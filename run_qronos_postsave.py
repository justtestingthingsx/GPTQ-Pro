#!/usr/bin/env python3
"""Post-save provenance fix for Qronos arms.

The Qronos dispatch rides the `gptaq` plumbing marker (see
gptq-pro-qronos.patch), so a freshly saved artifact's
quantization_config.meta records a `gptaq` entry it never ran. This script
rewrites the saved config.json to state the truth:

  meta.gptaq  -> removed
  meta.qronos -> {"percdamp": <value>, "paper": "arXiv:2505.11695",
                  "port": "target-quant/qronos qronos_gptqmodel.py"}

Run it IMMEDIATELY after the quant script, on the output dir, BEFORE upload:

  python run_qronos_postsave.py --artifact /out/Qwen3.8-27B-QronosArm \
      --percdamp 1e-5

It refuses to run if the artifact does not look like a Qronos arm (the run
report must say "solver": "qronos") so it can never mislabel a real GPTAQ
artifact. An artifact whose meta still says gptaq but whose run log shows
`Qronos ENGAGED` lines has NOT been through this step and must not ship.
"""

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", required=True, help="saved artifact dir")
    ap.add_argument("--percdamp", type=float, default=1e-5)
    args = ap.parse_args()

    root = Path(args.artifact)
    cfg_path = root / "config.json"
    report_path = root / "qwen3_5_27b_preflight.json"

    if not cfg_path.is_file():
        print(f"ERROR: {cfg_path} not found", file=sys.stderr)
        return 2

    # Guard: only rewrite artifacts whose own run report claims qronos.
    if not report_path.is_file():
        print(f"ERROR: {report_path} not found — cannot verify this is a "
              "Qronos arm; refusing to rewrite provenance", file=sys.stderr)
        return 2
    report = json.loads(report_path.read_text())
    if report.get("solver") != "qronos":
        print(f"ERROR: run report says solver={report.get('solver')!r}, not "
              "'qronos'; refusing to rewrite provenance", file=sys.stderr)
        return 2

    def _rewrite_meta(meta: dict) -> dict:
        meta.pop("gptaq", None)
        meta["qronos"] = {
            "percdamp": args.percdamp,
            "paper": "arXiv:2505.11695",
            "port": "target-quant/qronos qronos_gptqmodel.py",
        }
        meta["act_group_aware"] = False
        if "hessian" in meta:
            meta["hessian"] = {"note": "not honoured by qronos solver "
                                       "(process_batch overridden)"}
        if "fallback" in meta and isinstance(meta["fallback"], dict):
            meta["fallback"]["note"] = (
                "live only as the zero/under-threshold entry fallback "
                "under qronos")
        return meta

    cfg = json.loads(cfg_path.read_text())
    qc = cfg.get("quantization_config")
    if qc is None:
        print("ERROR: no quantization_config in config.json", file=sys.stderr)
        return 2
    meta = qc.get("meta", {})

    removed = "gptaq" in meta
    # _rewrite_meta covers the full normalisation (gptaq→qronos provenance +
    # the inert flags: act_group_aware False, hessian/fallback annotations;
    # activation_weighted_mse stays AS-IS — it IS honoured by the solver).
    qc["meta"] = _rewrite_meta(meta)
    cfg["quantization_config"] = qc
    cfg_path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")

    # Review-final F1: the SAME meta payload is written to a second file,
    # quantize_config.json — and the loader checks THAT file first
    # (config.py QUANT_CONFIG_FILENAME_COMPAT order). Rewrite it too.
    qcfg_path = root / "quantize_config.json"
    if qcfg_path.is_file():
        qcfg = json.loads(qcfg_path.read_text())
        if isinstance(qcfg.get("meta"), dict):
            qcfg["meta"] = _rewrite_meta(qcfg["meta"])
            qcfg_path.write_text(json.dumps(qcfg, indent=2) + "\n",
                                 encoding="utf-8")

    # Final verification pass: NO file may still claim gptaq (F1's exit gate).
    bad = []
    for p in (cfg_path, qcfg_path):
        if not p.is_file():
            continue
        payload = json.loads(p.read_text())
        metas = [payload.get("meta"),
                 (payload.get("quantization_config") or {}).get("meta")]
        if any(isinstance(m, dict) and "gptaq" in m for m in metas):
            bad.append(p.name)
    if bad:
        print(f"ERROR: meta.gptaq survives in {bad} — artifact must not ship",
              file=sys.stderr)
        return 3

    print(f"[ok] provenance rewritten in config.json + quantize_config.json: "
          f"gptaq removed={removed}, qronos added (percdamp={args.percdamp}), "
          f"inert flags normalised, verification pass clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
