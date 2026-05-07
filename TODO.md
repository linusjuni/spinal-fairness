# TODO

## When Dataset002 + Dataset003 training finishes

```bash
# Predict + postprocess (repeat for dataset 3)
bash jobs/submit_predict.sh 2
bash jobs/submit_predict.sh 3
# wait for predict jobs
bash jobs/ensemble_and_postprocess.sh 2
bash jobs/ensemble_and_postprocess.sh 3
```

Then run fairness analysis — see `src/fairness/README.md`.
