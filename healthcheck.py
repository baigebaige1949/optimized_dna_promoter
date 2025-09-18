from __future__ import annotations
import argparse, importlib, json, os, re, sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parent
PKG = ROOT.name
PARENT = ROOT.parent

def ensure_path():
    if str(PARENT) not in sys.path:
        sys.path.insert(0, str(PARENT))

def check_init(autofix: bool) -> Dict[str, Any]:
    sub = ["core","models","data","training","evaluation","generation","conditions","utils","version_control","examples","config"]
    created, missing = [], []
    for d in [ROOT] + [ROOT/p for p in sub if (ROOT/p).is_dir()]:
        ini = d/"__init__.py"
        if not ini.exists():
            if autofix: ini.write_text("", encoding="utf-8"); created.append(str(ini))
            else: missing.append(str(ini))
    return {"created":created,"missing":missing,"ok":len(missing)==0}

def patch_core_alias(autofix: bool)->Dict[str,Any]:
    core_init = ROOT/"core"/"__init__.py"
    if not core_init.exists(): return {"ok":False,"message":"core/__init__.py 不存在","patched":False}
    s = core_init.read_text(encoding="utf-8")
    bad = re.search(r"from\s+\.\s*predictor_model\s+import\s+PredictorModel", s)
    has_imp = "from .predictor_model import PromoterPredictor" in s
    has_alias = re.search(r"PredictorModel\s*=\s*PromoterPredictor", s)
    if has_imp and has_alias: return {"ok":True,"patched":False,"message":"已存在别名"}
    if not (bad or not (has_imp and has_alias)):
        # 尝试运行态验证
        try:
            ensure_path()
            from optimized_dna_promoter.core import PredictorModel, PromoterPredictor  # type: ignore
            return {"ok": PredictorModel is PromoterPredictor, "patched":False,"message":"运行态已OK"}
        except Exception:
            pass
    if not autofix: return {"ok":False,"patched":False,"message":"需要修补（加 --autofix）"}
    bak = core_init.with_suffix(".py.bak_healthcheck")
    bak.write_text(s, encoding="utf-8")
    s = re.sub(r"(^\s*from\s+\.\s*predictor_model\s+import\s+PredictorModel.*?$)",
               r"# \1  # disabled by healthcheck", s, flags=re.M)
    if "from .predictor_model import PromoterPredictor" not in s:
        s += "\nfrom .predictor_model import PromoterPredictor\n"
    if "PredictorModel = PromoterPredictor" not in s:
        s += "PredictorModel = PromoterPredictor  # compat alias\n"
    core_init.write_text(s, encoding="utf-8")
    return {"ok":True,"patched":True,"message":f"已修补，备份 {bak.name}"}

def dep_check()->Dict[str,Any]:
    req = [("torch","version"),("transformers","__version__"),("numpy","__version__"),
           ("sklearn","__version__"),("Bio","__version__"),("pandas","__version__"),
           ("yaml","__version__"),("tqdm","__version__"),("matplotlib","__version__")]
    ver, missing = {}, []
    for m, attr in req:
        try:
            mod = importlib.import_module(m)
            ver[m] = str(getattr(mod, attr, "unknown"))
        except Exception as e:
            ver[m] = f"NOT FOUND: {e}"
            missing.append(m)
    return {"versions":ver,"missing":missing,"ok":len(missing)==0}

def get_cfg():
    try:
        m = importlib.import_module(f"{PKG}.config.transformer_config")
        return getattr(m,"TransformerPredictorConfig")
    except Exception:
        m = importlib.import_module(f"{PKG}.config.model_config")
        return getattr(m,"PredictorModelConfig")

def build_universal():
    rep = {"built":False,"error":None,"cls":None}
    try:
        factory = importlib.import_module(f"{PKG}.models.model_factory")
        Cfg = get_cfg(); cfg = Cfg()
        pred = factory.PredictorModelFactory.create_predictor("transformer", cfg, return_universal=True)
        rep.update(built=True, cls=type(pred).__name__)
        return pred, rep
    except Exception as e:
        rep["error"] = f"{type(e).__name__}: {e}"
        return None, rep

def exercise(pred)->Dict[str,Any]:
    out = {"ok":True}
    seqs = ["ATGCGTACGT","TATAATGGCC","ATATATATAT"]
    try: out["predict_strength"] = pred.predict_strength(seqs[:2])
    except Exception as e: out["predict_strength"]=f"FAIL: {e}"; out["ok"]=False
    try: out["predict_batch"] = pred.predict_batch(seqs, batch_size=2)
    except Exception as e: out["predict_batch"]=f"FAIL: {e}"; out["ok"]=False
    try: out["get_feature_importance"] = pred.get_feature_importance(seqs[:1])
    except Exception as e: out["get_feature_importance"]=f"SKIP/FAIL: {e}"
    try: out["get_model_summary"] = pred.get_model_summary()
    except Exception as e: out["get_model_summary"]=f"FAIL: {e}"; out["ok"]=False
    try: out["benchmark_inference"] = pred.benchmark_inference(seqs, num_runs=2)
    except Exception as e: out["benchmark_inference"]=f"FAIL: {e}"
    return out

def smoke_demo()->Dict[str,Any]:
    try:
        importlib.import_module(f"{PKG}.examples.conditional_diffusion_demo")
        return {"ok":True,"import_ok":True}
    except Exception as e:
        return {"ok":False,"import_ok":f"{type(e).__name__}: {e}"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--autofix", action="store_true")
    ap.add_argument("--smoke-import-demo", action="store_true")
    ap.add_argument("--json-report", action="store_true")
    args = ap.parse_args()

    ensure_path()
    report: Dict[str,Any] = {"project_root":str(ROOT)}

    report["init_files"] = check_init(args.autofix)
    report["core_alias_patch"] = patch_core_alias(args.autofix)
    report["deps"] = dep_check()

    # 运行时别名校验
    try:
        from optimized_dna_promoter.core import PredictorModel, PromoterPredictor  # type: ignore
        report["alias_runtime"] = {"ok": PredictorModel is PromoterPredictor}
    except Exception as e:
        report["alias_runtime"] = {"ok":False,"error":str(e)}

    pred, build_info = build_universal()
    report["predictor_build"] = build_info
    report["predictor_api"] = exercise(pred) if pred else {"ok":False,"error":"构建失败"}

    if args.smoke_import_demo:
        report["demo_smoke"] = smoke_demo()

    oks = [
        report["init_files"]["ok"], report["core_alias_patch"]["ok"], report["deps"]["ok"],
        report["alias_runtime"]["ok"], report["predictor_build"]["built"], report["predictor_api"]["ok"]
    ]
    if args.smoke_import_demo: oks.append(report["demo_smoke"]["ok"])
    report["overall_ok"] = all(oks)

    print("="*60)
    print(f"[healthcheck] root: {ROOT}")
    print("- 包结构:", "OK" if report["init_files"]["ok"] else "ISSUE")
    print("- core 别名:", "OK" if report["core_alias_patch"]["ok"] else "ISSUE")
    print("- 依赖:", "OK" if report["deps"]["ok"] else f"缺失 -> {report['deps']['missing']}")
    print("- 运行时别名:", "OK" if report["alias_runtime"]["ok"] else "FAIL")
    print("- 预测器构建:", "OK" if report["predictor_build"]["built"] else report["predictor_build"]["error"])
    if pred:
        api = report["predictor_api"]
        print("- 接口: predict_strength:", api.get("predict_strength"))
        print("       predict_batch:", api.get("predict_batch"))
        print("       get_feature_importance:", api.get("get_feature_importance"))
        print("       get_model_summary:", api.get("get_model_summary"))
        print("       benchmark_inference:", api.get("benchmark_inference"))
    if args.smoke_import_demo:
        print("- demo 导入:", "OK" if report["demo_smoke"]["ok"] else report["demo_smoke"]["import_ok"])
    print("-"*60)
    print("OVERALL:", "PASS ✅" if report["overall_ok"] else "FAIL ❌")

    if args.json_report:
        (ROOT/"health_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[healthcheck] JSON -> {ROOT/'health_report.json'}")

if __name__ == "__main__":
    main()
