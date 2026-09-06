#!/usr/bin/env python
"""Collect final_score + sub-scores for every finished rnv2 run into one ranked table.

Reads the probe result JSONs (not the folder listing) so rows survive checkpoint rotation.
"""
import json, glob, os, sys

FIELDS = [("final_score", "final"), ("croma_mean", "croma"), ("robustness_mean", "robust"),
          ("classification_mean_f1", "cls_f1"), ("seg_mean_f1", "seg_f1"),
          ("slide_mean_auc", "slide_auc"), ("survival_mean_cindex", "surv_c")]

rows = []
for d in sorted(glob.glob("/data/dojo/nanopath/main/rnv2-*")):
    res = sorted(glob.glob(f"{d}/probe/results/*.json"))
    if not res:
        continue
    m = json.load(open(res[-1])).get("metrics", {})
    if "final_score" not in m:
        continue
    rows.append((os.path.basename(d), [m.get(k) for k, _ in FIELDS]))

if not rows:
    print("no finished rnv2 runs yet")
    sys.exit()

base = next((v[0] for n, v in rows if "baseline" in n), None)
rows.sort(key=lambda r: -(r[1][0] or 0))
hdr = f"{'run':<30}" + "".join(f"{lbl:>11}" for _, lbl in FIELDS) + ("      delta" if base else "")
print(hdr); print("-" * len(hdr))
for name, vals in rows:
    line = f"{name:<30}" + "".join(f"{v:>11.4f}" if isinstance(v, float) else f"{'--':>11}" for v in vals)
    if base and vals[0] is not None:
        d = vals[0] - base
        line += f"   {d:+.4f}" + ("  <-- baseline" if "baseline" in name else "")
    print(line)
if base:
    print(f"\nbaseline = {base:.4f};  seed-noise band on this harness is ~0.007-0.009 (ledger finding 10),")
    print("so treat |delta| < 0.010 as unresolved rather than as an effect.")
