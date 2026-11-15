"""E4 - deployment cost of the trained models (docs/phase3_protocol.md §9.5).

For each experiment, the checkpoint of seed 42 is loaded (latency does not depend on
the weight values; the checkpoint guarantees the exact trained architecture).

Protocol: CPU fp32 with 4 threads (batch 1 and 32), GPU fp32 and AMP fp16 (batch 1
and 32); warm-up 50 + 200 timed iterations (CPU batch 32: 10 + 50); median and IQR;
the whole benchmark is repeated 3 times with the model order alternated, and the
median of the 3 medians is reported. Must run on mains power.

Usage:
    python scripts/benchmark_models.py
    python scripts/benchmark_models.py --experiments custom_cnn resnet50_ft --quick --out <dir>   # smoke test
Outputs: reports/benchmarks/phase3/benchmark.json, benchmark.md (new folder; nothing is overwritten
unless --force on this benchmark's own folder).
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from ivai.config import ENGINE_DIR, REPORTS_DIR, load_data_config
from ivai.data.dataset import NEUDataset, build_transforms, resolve_normalization
from ivai.evaluation.benchmark import (count_macs, measure_latency, measure_preprocessing, on_ac_power,
                                       state_dict_size_mb)
from ivai.evaluation.comparison import load_experiment, model_from_run
from ivai.models.analysis import count_parameters

DEFAULT_EXPERIMENTS = ["custom_cnn", "resnet50_ft", "mobilenetv2_ft", "mobilenetv2_scratch", "resnet50_probe",
                       "mobilenetv2_probe"]
CPU_THREADS = 4


def settings(quick: bool) -> list[dict]:
    s = [
        {"device": "cpu", "amp": False, "batch": 1, "warmup": 50, "iters": 200},
        {"device": "cpu", "amp": False, "batch": 32, "warmup": 10, "iters": 50},
    ]
    if torch.cuda.is_available():
        for amp in (False, True):
            for b in (1, 32):
                s.append({"device": "cuda", "amp": amp, "batch": b, "warmup": 50, "iters": 200})
    if quick:
        for x in s:
            x["warmup"], x["iters"] = 2, 5
    return s


def key(st: dict) -> str:
    return f"{st['device']}_{'amp' if st['amp'] else 'fp32'}_b{st['batch']}"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiments", nargs="+", default=DEFAULT_EXPERIMENTS)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", type=Path, default=REPORTS_DIR / "benchmarks" / "phase3")
    ap.add_argument("--quick", action="store_true", help="few iterations (smoke test only, not a measurement)")
    ap.add_argument("--allow-battery", action="store_true")
    ap.add_argument("--force", action="store_true", help="overwrite this benchmark's own output folder")
    args = ap.parse_args()

    power = on_ac_power()
    if power is False and not args.allow_battery:
        raise SystemExit("On battery: latency would be throttled. Plug in the charger (or --allow-battery for a smoke test).")
    if (args.out / "benchmark.json").exists() and not args.force:
        raise SystemExit(f"{args.out / 'benchmark.json'} exists; refusing to overwrite (use --force or another --out).")
    args.out.mkdir(parents=True, exist_ok=True)
    classes = load_data_config().classes

    models = {}
    for name in args.experiments:
        exp = load_experiment(name, classes)
        if exp is None or args.seed not in exp.runs:
            print(f"skip {name}: no run for seed {args.seed}")
            continue
        model, inp = model_from_run(exp.runs[args.seed])
        ckpt = ENGINE_DIR.parent / exp.runs[args.seed]["checkpoint"]
        shape = (inp["channels"], 200, 200)
        mean, std = resolve_normalization(inp["normalization"])
        models[name] = {"model": model, "shape": shape, "transform": build_transforms(False, mean, std, channels=inp["channels"]),
                        "info": {"parameters": count_parameters(model), "gmac_200px": count_macs(model, (1, *shape)) / 1e9,
                                 "state_dict_fp32_mb": state_dict_size_mb(model),
                                 "checkpoint_file_mb": ckpt.stat().st_size / 2**20, "input": inp,
                                 "checkpoint": exp.runs[args.seed]["checkpoint"]}}
    if not models:
        raise SystemExit("no model to benchmark")

    image = NEUDataset("val", transform=None).images[0]   # one validation image (uint8), for preprocessing time
    sts = settings(args.quick)
    raw = {n: {key(s): [] for s in sts} for n in models}
    prep = {n: [] for n in models}
    order = list(models)
    for r in range(args.repeats):
        seq = order if r % 2 == 0 else order[::-1]          # alternate the model order
        print(f"repeat {r + 1}/{args.repeats}: {seq}")
        for n in seq:
            m = models[n]
            w, it = (2, 5) if args.quick else (50, 200)
            prep[n].append(measure_preprocessing(m["transform"], image, w, it))
            for s in sts:
                res = measure_latency(m["model"], (s["batch"], *m["shape"]), s["device"], s["warmup"], s["iters"],
                                      amp=s["amp"], num_threads=CPU_THREADS if s["device"] == "cpu" else None)
                raw[n][key(s)].append(res)
            m["model"].cpu()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    results = {}
    for n in models:
        lat = {k: {"median_of_medians_ms": float(np.median([x["median_ms"] for x in v])),
                   "median_iqr_ms": float(np.median([x["iqr_ms"] for x in v])),
                   "throughput_img_s": float(np.median([x["throughput_img_s"] for x in v])), "repeats": v}
               for k, v in raw[n].items()}
        results[n] = {**models[n]["info"], "latency": lat,
                      "preprocessing_ms": float(np.median([p["median_ms"] for p in prep[n]]))}

    out = {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "quick_smoke_test": args.quick,
           "protocol": "docs/phase3_protocol.md §9.5", "seed_checkpoint": args.seed, "repeats": args.repeats,
           "environment": {"cpu": platform.processor(), "cpu_threads_used": CPU_THREADS, "torch": torch.__version__,
                           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                           "on_ac_power": power},
           "settings": sts, "models": results}
    (args.out / "benchmark.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    cols = [key(s) for s in sts]
    md = [f"# Phase 3 benchmark{' (QUICK SMOKE TEST - not a measurement)' if args.quick else ''}", "",
          f"Median of {args.repeats} medians (ms). CPU: {CPU_THREADS} threads, fp32. AC power: {power}.", "",
          "| Model | Params | GMAC | fp32 MB | Prep. ms | " + " | ".join(cols) + " |",
          "|---|---|---|---|---|" + "---|" * len(cols)]
    for n, r in results.items():
        md.append(f"| {n} | {r['parameters']['total']:,} | {r['gmac_200px']:.3f} | {r['state_dict_fp32_mb']:.1f} | "
                  f"{r['preprocessing_ms']:.3f} | " + " | ".join(f"{r['latency'][c]['median_of_medians_ms']:.2f}" for c in cols) + " |")
    (args.out / "benchmark.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
