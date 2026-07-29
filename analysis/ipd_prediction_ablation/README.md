# Matched-rollout prediction ablation

After the control run has logged 100 learner updates, generate the comparison:

```bash
python analysis/ipd_prediction_ablation/compare_matched_runs.py \
  --control-run ub-ipd-no-predict-matched-100-19095989
```

The script compares updates 1–100 against prediction run
`ub-ipd-typed-recent-200-18994851` and writes `matched_comparison.json`.
The shared training unit is 128 communication and 128 decision samples per
role/update, each with an absolute loss coefficient of 1/3.
