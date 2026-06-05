# TODO

## Training status (2026-06-05)

- Dataset002 (gold): all 10 models done (2d + 3d_fullres, folds 0-4)
- Dataset003 (silver) 2d: all 5 folds done
- Dataset003 (silver) 3d_fullres:
  - fold 0: checkpoint done, validation job submitted (28602612, gpul40s)
  - fold 1: training in progress (job 28602550, gpua100)
  - fold 2, 3: done
  - fold 4: training in progress (job 28602611, gpua100)

## When Dataset003 training finishes

```bash
# Predict + postprocess for gold and silver
bash jobs/submit_predict.sh 2
bash jobs/submit_predict.sh 3
# wait for predict jobs
bash jobs/ensemble_and_postprocess.sh 2
bash jobs/ensemble_and_postprocess.sh 3
```

Then run fairness analysis — see `src/fairness/README.md`.
