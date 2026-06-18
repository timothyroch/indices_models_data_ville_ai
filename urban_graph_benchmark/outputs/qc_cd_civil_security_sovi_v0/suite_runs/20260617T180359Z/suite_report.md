# Québec CD civil-security / SoVI benchmark suite run

Started: **2026-06-17T18:03:59+00:00**
Ended: **2026-06-17T18:04:55+00:00**

## Summary

- Steps selected: **8**
- Success: **0**
- Skipped by resume: **0**
- Failed: **1**
- Manifest: `urban_graph_benchmark/outputs/qc_cd_civil_security_sovi_v0/suite_runs/20260617T180359Z/suite_manifest.json`

## Step status

| Status | Step | Duration seconds | Return code | Missing expected outputs | Logs |
|---|---|---:|---:|---|---|
| failed failed_missing_outputs | `10_b1_sovi_direct_validation` | 56.21 | 0 | `urban_graph_benchmark/outputs/qc_cd_civil_security_sovi_v0/baselines/B1_sovi_direct_validation/metrics.csv`<br>`urban_graph_benchmark/outputs/qc_cd_civil_security_sovi_v0/baselines/B1_sovi_direct_validation/metadata.json` | `logs/10_b1_sovi_direct_validation.stdout.log` / `logs/10_b1_sovi_direct_validation.stderr.log` |

## Commands

### 10_b1_sovi_direct_validation

```bash
/home/tim/Documents/ville_ai/indices/transformation/.venv/bin/python urban_graph_benchmark/scripts/10_run_qc_cd_b1_sovi_direct_validation.py --n-permutations 100 --n-bootstraps 100
```
