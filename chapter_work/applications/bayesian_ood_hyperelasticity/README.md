# bayesian_ood_hyperelasticity

Residual correction wired into the existing hyperelasticity Bayesian-inversion
pipeline (`survey_work/applications/bayesian_inverse_problem_hyperelasticity`,
`survey_work/problems/hyperelasticity`), for `sec:bayesian`, "Inference using
a shifted prior".

## Changes to shared code

- `survey_work/problems/hyperelasticity/hyperelasticityModel.py`: added
  `HyperelasticityModel.residual_correct(m, u_tilde)` -- one Newton step from
  a supplied predicted state (`max_it=1` on the existing `NewtonSolver`).
  `eq:corrected_state` applied to the hyperelastic problem.
- `src/mcmc/correctedSurrogateModel.py`: `CorrectedSurrogateModel` wraps a
  `SurrogateModel`/`SurrogateModelFNO`; `solveFwd(w)` returns the prediction
  after one `residual_correct` step. Drop into `MCMC.surrogate_models` under
  a key like `'DeepONet-corrected'`.

`survey_work/` results are untouched.

## Files

- `generate_ground_truth_ood.py` -- ground truth with w pushed further from
  the training distribution than `Generate_GroundTruth.ipynb` (larger
  multi-mode amplitude + prior-noise scale; see module docstring).
- `check_correction.py` -- run first. No MCMC; prints raw vs. corrected
  state error at a few distribution-shift levels.
- `run_bayesian_inversion_ood.py` -- pCN-MCMC per neural operator (reference
  FE / raw NOP / corrected NOP), saves `summary.json` (wall time,
  posterior-mean error in w and m). `--quick` for a short run; omit for the
  article-scale run (10,000 samples + 500 burn-in).
- `process_ood_results.py` / `merge_summaries.py` -- post-process an MCMC
  sweep into the ground-truth panel and per-surrogate posterior-mean figures.

## How to run

```bash
conda activate neuralopv2
cd chapter_work/applications/bayesian_ood_hyperelasticity
python generate_ground_truth_ood.py
python check_correction.py
python run_bayesian_inversion_ood.py --quick   # checks the driver runs end to end
python run_bayesian_inversion_ood.py           # full run; reference (FE) chain is slow
```

The reference chain calls the FE hyperelasticity solve for `n_samples +
n_burnin` proposals, so it's the long pole -- hours at article scale. The
corrected chains are one tangent solve per proposal.

## Status

Results are already in the draft: `sections/07_bayesian.tex` ("Inference
using a shifted prior") and `figures/bayesian_hyperelasticity_ood/`.
Re-run only if the numbers need to change.
