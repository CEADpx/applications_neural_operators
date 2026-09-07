# bayesian_ood_hyperelasticity

Residual correction, minimally wired into this repository's existing hyperelasticity
Bayesian-inversion pipeline (`survey_work/applications/bayesian_inverse_problem_hyperelasticity`
and `survey_work/problems/hyperelasticity`), for the book chapter's Bayesian-inference
application (`sec:bayesian`, "Inference using a shifted prior").

Originally developed under `res_corr_work/` before that folder was merged into
`chapter_work/`; nothing about the pipeline itself changed in the move (see the fixed
`ROOT` path at the top of each script -- it now resolves three directories up, since this
folder sits at `chapter_work/applications/bayesian_ood_hyperelasticity`).

## What changed in the shared code (minimal, additive only)

- `survey_work/problems/hyperelasticity/hyperelasticityModel.py`: added
  `HyperelasticityModel.residual_correct(m, u_tilde)`. One Newton step of the existing
  hyperelastic residual form, taken from a supplied predicted state instead of from zero
  (`max_it=1` on the module's own, already-validated `NewtonSolver`). This is
  `eq:corrected_state` from `sec:correction` of the chapter, applied to the hyperelastic
  problem. Same pattern as the corrector in the companion `agent_neural_operator` repo's
  `Hyperelastic2D.residual_based_correction` / `single_newton_step`.
- `src/mcmc/correctedSurrogateModel.py`: small `CorrectedSurrogateModel` class. Wraps
  an existing `SurrogateModel`/`SurrogateModelFNO` instance; `solveFwd(w)` returns the
  neural-operator prediction after one `residual_correct` step. Drop into
  `MCMC.surrogate_models` under a new key (e.g. `'DeepONet-corrected'`) and set
  `mcmc.surrogate_to_use` to that key -- no changes to `MCMC`, `Tracer`, or the neural
  operator training code were needed.

Nothing in `survey_work/` (the article's original results) is modified.

## What's here

- `generate_ground_truth_ood.py` -- ground truth whose w field is pushed further from the
  training distribution than
  `survey_work/applications/bayesian_inverse_problem_hyperelasticity/Generate_GroundTruth.ipynb`
  (larger multi-mode amplitude + larger prior-noise contribution; see module docstring for
  the reasoning, including why the physical transform m(w) is deliberately kept identical
  to training rather than also shifted).
- `smoke_test_correction.py` -- **run this first.** No MCMC; single forward evaluations at
  a few distribution-shift levels, printing raw vs. corrected state error against the FE
  solution. Corrected error should be well below raw error and the gap should widen with
  distribution shift. If not, something in the Newton-step plumbing needs debugging before
  any MCMC chain is trustworthy.
- `run_bayesian_inversion_ood.py` -- runs pCN-MCMC three ways per neural operator
  (reference FE / raw NOP / corrected NOP), saves a `summary.json` with wall time and
  posterior-mean error in `w` and `m` for each chain. `--quick` for a fast smoke test of
  the MCMC driver itself (small sample count); omit it for the article-scale run (10,000
  samples + 500 burn-in, matching `BayesianInversion.ipynb`'s convention) used for the
  chapter numbers.
- `process_ood_results.py` / `merge_summaries.py` -- post-process an MCMC sweep into the
  ground-truth panel and per-surrogate posterior-mean figures used by the chapter.

## How to run

```bash
conda activate neuralopv2
cd chapter_work/applications/bayesian_ood_hyperelasticity
python generate_ground_truth_ood.py
python smoke_test_correction.py          # sanity check -- read the [OK]/[CHECK ME] flags
python run_bayesian_inversion_ood.py --quick   # ~minutes, checks the MCMC driver runs end to end
python run_bayesian_inversion_ood.py           # full run; reference (FE) chain is the slow one
```

The reference chain calls the FE hyperelasticity solve (~2s/sample from the existing
in-distribution notebook's timing) for `n_samples + n_burnin` proposals, so it is the long
pole -- expect it to take hours at article scale. The corrected chains are one tangent
solve per proposal instead of a full incremental Newton solve, so noticeably cheaper than
the reference but more expensive than the raw surrogate.

## Status

Results from this pipeline are already incorporated into the draft: `sections/07_bayesian.tex`
("Inference using a shifted prior") and `figures/bayesian_hyperelasticity_ood/` in the
chapter draft. Re-run and re-process here only if the numbers need to change.
