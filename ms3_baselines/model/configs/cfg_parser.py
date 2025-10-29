# parse_cfg.py
# -*- coding: utf-8 -*-
import argparse, yaml, ast, copy, os, re, sys
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional

# ============ 向上查找 YAML（未传 --cfg 时自动寻找） ============
def find_yaml_files(start_dir: str) -> List[str]:
    cur_dir = os.path.abspath(start_dir)
    root_dir = os.path.abspath(os.sep)
    while True:
        files = [f for f in os.listdir(cur_dir) if f.endswith(('.yaml', '.yml'))]
        if files:
            return [os.path.join(cur_dir, f) for f in files]
        if cur_dir == root_dir:
            break
        cur_dir = os.path.dirname(cur_dir)
    return []

# ============ 工具函数：合并 / 覆盖 / 类型推断 ============
def _merge(a: Dict, b: Dict) -> Dict:
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out

def _set_by_dotpath(d: Dict, key: str, value: Any):
    ks = key.split(".")
    cur = d
    for k in ks[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[ks[-1]] = value

def _infer_type(s: str):
    try:
        return ast.literal_eval(s)
    except Exception:
        low = s.lower()
        if low == "true": return True
        if low == "false": return False
        if low in ("none", "null"): return None
        return s

# ============ 占位符展开：${here}/${cwd}/${home}/${env:VAR} 与 ~ ============
def _path_context():
    main_file = getattr(sys.modules.get("__main__"), "__file__", None)
    here = Path(main_file).resolve().parent if main_file else Path(os.getcwd()).resolve()
    return {
        "here": str(here),
        "cwd": str(Path(os.getcwd()).resolve()),
        "home": str(Path.home().resolve()),
    }

def _expand_placeholders(obj, mapping):
    if isinstance(obj, dict):
        return {k: _expand_placeholders(v, mapping) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_placeholders(v, mapping) for v in obj]
    if isinstance(obj, str):
        s = os.path.expanduser(obj)  # 处理 ~
        for k, v in mapping.items():
            s = s.replace(f"${{{k}}}", v)
        s = re.sub(r"\$\{env:([A-Za-z_][A-Za-z0-9_]*)\}", lambda m: os.getenv(m.group(1), ""), s)
        return s
    return obj

# ============ 模板渲染：{date:%Y..} 与 {a.b.c} / {run} ============
def _get_by_dot(cfg: Dict, dot: str):
    cur = cfg
    for k in dot.split("."):
        cur = cur[k]
    return cur

def _format_with_cfg(tpl: str, cfg: Dict) -> str:
    if tpl is None:
        return None
    s = tpl

    # {date:%Y%m%d-%H%M%S}
    def repl_date(m):
        fmt = m.group(1)
        return datetime.now().strftime(fmt)
    s = re.sub(r"\{date:([^}]+)\}", repl_date, s)

    # {a.b.c} 或 {run}（run 代表 run.name）
    def repl_dot(m):
        key = m.group(1)
        if key == "run":
            return str(cfg.get("run", {}).get("name", ""))
        return str(_get_by_dot(cfg, key))
    s = re.sub(r"\{([A-Za-z0-9_.]+)\}", repl_dot, s)
    return s

def _slug(s: str) -> str:
    return re.sub(r"[^\w\-\.]+", "_", s).strip("_")

def _probe_cfg_path(base_dir: Path, name: str) -> Optional[Path]:
    """在 base_dir 及 base_dir/configs 下按 name, name.yaml, name.yml 顺序查找"""
    candidates = [name, f"{name}.yaml", f"{name}.yml"]
    for rel in candidates:
        p = base_dir / rel
        if p.exists(): return p
    cfg_dir = base_dir / "configs"
    for rel in candidates:
        p = cfg_dir / rel
        if p.exists(): return p
    return None

# ============ 主入口：解析 cfg，渲染 run/ckpt，并创建目录 ============
def parse_cfg(cfg_name: Optional[str]=None) -> Dict:
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", nargs="+", required=False, help="一个或多个 YAML，按顺序合并（可给出完整路径）")
    p.add_argument("overrides", nargs="*", help="dotlist 覆盖，如 train.lr=1e-4 model.latent_dim=64 run.tpl='...'")
    args = p.parse_args()

    # ---------- 选择配置文件列表 ----------
    if args.cfg is None:
        this_dir = Path(__file__).resolve().parent

        if cfg_name:  # 显式指定 cfg 名称：先 base，再指定；找不到指定就用“默认配置”
            base_p  = _probe_cfg_path(this_dir, "base")
            named_p = _probe_cfg_path(this_dir, cfg_name)

            if named_p is None:
                # 明确提示：未找到，使用默认配置
                if base_p:
                    print(f"\033[33m[Warn] 未找到指定配置 '{cfg_name}'，将使用默认配置: {base_p}\033[0m")
                    args.cfg = [str(base_p)]
                    print(f"\033[33m[Auto] 合并顺序（后者覆盖前者）: {args.cfg}\033[0m")
                else:
                    auto = find_yaml_files(str(this_dir))
                    if not auto:
                        raise FileNotFoundError(
                            f"未找到指定配置 '{cfg_name}'，且未找到 base.yaml；也未在路径上发现任何 yaml"
                        )
                    print(f"\033[33m[Warn] 未找到指定配置 '{cfg_name}'，也未找到 base；改用自动发现的配置\033[0m")
                    args.cfg = auto
                    print(f"\033[33m[Auto] 合并顺序（后者覆盖前者）: {args.cfg}\033[0m")
            else:
                # 找到 named：按 [base, named] 顺序（后者覆盖前者）
                paths: List[str] = []
                if base_p:
                    paths.append(str(base_p))
                # 避免 base 与 named 是同一文件时重复
                if (not base_p) or (named_p.resolve() != base_p.resolve()):
                    paths.append(str(named_p))
                args.cfg = paths
                print(f"\033[33m[Auto] 合并顺序（后者覆盖前者）: {args.cfg}\033[0m")

        else:
            # 未指定 cfg_name：只用 base；找不到才向上自动搜索
            base_p = _probe_cfg_path(this_dir, "base")
            if base_p:
                args.cfg = [str(base_p)]
                print(f"\033[33m[Auto] 使用默认配置: {args.cfg}\033[0m")
            else:
                auto = find_yaml_files(str(this_dir))
                if not auto:
                    raise FileNotFoundError("未找到 base.yaml，且向上未发现任何 yaml，请显式传 --cfg")
                args.cfg = auto
                print(f"\033[33m[Auto] 使用自动发现的配置（后者覆盖前者）: {args.cfg}\033[0m")

    # ---------- 合并 YAML（后者覆盖前者） ----------
    cfg: Dict = {}
    for path in args.cfg:
        with open(path, "r") as f:
            y = yaml.safe_load(f) or {}
        cfg = _merge(cfg, y)

    # ---------- dotlist 覆盖 ----------
    for ov in args.overrides:
        if "=" not in ov:
            raise ValueError(f"Bad override: {ov}")
        k, v = ov.split("=", 1)
        _set_by_dotpath(cfg, k, _infer_type(v))

    # ---------- 展开 ${...} ----------
    cfg = _expand_placeholders(cfg, _path_context())

    # ---------- 渲染 run.name ----------
    run_tpl = cfg.get("run", {}).get("tpl") or "{date:%Y%m%d-%H%M%S}"
    run_name = _slug(_format_with_cfg(run_tpl, cfg))
    cfg.setdefault("run", {})["name"] = run_name
    cfg["run"]["id"] = run_name  # 同名别称

    # ---------- 渲染并回写 ckpt_root ----------
    root_tpl  = cfg.get("paths", {}).get("ckpt_root", "${here}/ckpt")
    root_fmt  = _format_with_cfg(root_tpl, cfg)
    root_fmt  = _expand_placeholders(root_fmt, _path_context())
    ckpt_root = Path(root_fmt).expanduser().resolve()
    cfg.setdefault("paths", {})["ckpt_root"] = str(ckpt_root)

    # ---------- 渲染 ckpt_dir（支持 ckpt_tpl 或 ckpt_path） ----------
    ckpt_tpl = cfg["paths"].get("ckpt_tpl", cfg["paths"].get("ckpt_path", None))
    if ckpt_tpl is None:
        ckpt_tpl = str(ckpt_root / "{run}")
    ckpt_dir_str = _format_with_cfg(ckpt_tpl, cfg)
    ckpt_dir_str = _expand_placeholders(ckpt_dir_str, _path_context())
    ckpt_dir     = Path(ckpt_dir_str).expanduser().resolve()
    cfg["paths"]["save_ckpt_dir"] = str(ckpt_dir)

    # ---------- 渲染 wandb_group ----------
    if "log" in cfg and "wandb_group" in cfg["log"]:
        cfg["log"]["wandb_group"] = _format_with_cfg(cfg["log"]["wandb_group"], cfg)

    return cfg

# 实例用法
if __name__ == "__main__":
    from configs.cfg_parser import parse_cfg
    # 给定配置名称cfg_name， 将会覆盖默认的 base 配置
    # 如果 cfg_name 为 None，则只使用 base 配置（找不到则自动搜索）
    cfg = parse_cfg(cfg_name = 'MDN')

    print("run.name =", cfg["run"]["name"])
    print("ckpt_root =", cfg["paths"]["ckpt_root"])
    print("save_ckpt_dir =", cfg["paths"].get("save_ckpt_dir"))
