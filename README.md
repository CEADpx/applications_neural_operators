# applications_neural_operators

Applications built on the [neural_operators](https://github.com/CEADpx/neural_operators) library: the neural-operator survey article and the book chapter on residual correction.

Each subfolder pins the commit or tag of `neural_operators` it was built against (see table below). Update the pin explicitly when adopting library changes; don't assume the latest library commit works with an existing subfolder.

## Environment

Same conda environment as `neural_operators` (`neuralopv2`); see that repo's [neuralop.yml](https://github.com/CEADpx/neural_operators/blob/main/neuralop.yml).

## Repository layout

| Folder | Built against `neural_operators` | Content |
|--------|-----------------------------------|---------|
| [survey_work](survey_work) | tag `v1.0.0` | Neural-operator survey article: Poisson, linear elasticity, hyperelasticity forward problems, NOP training, Bayesian inversion |
| [chapter_work](chapter_work) | tag `v1.0.0` | Book chapter: reaction-diffusion problem, topology optimization, residual-corrected Bayesian inference under distribution shift |

`v1.0.0` is the first release of `neural_operators` as an installable package (`pip install -e .` from that repo, or pin the tag directly: `pip install "neural_operators @ git+https://github.com/CEADpx/neural_operators.git@v1.0.0"`). Both folders were ported off the pre-package flat `src/{pde,prior,plotting,...}` layout (`survey26_v2`-era) onto this tag in the same pass; update both rows together if either folder adopts a newer library tag, since a folder half-migrated to a new API is worse than one pinned to an old one.

Trained artifacts (`model.pkl`, MCMC `tracer.pkl`) saved under the old flat layout won't unpickle under `v1.0.0` (Python pickles the class's module path at save time, and the old flat modules no longer exist) -- retrain rather than trying to load them. Fresh data/samples (`.npz`) are unaffected; only pickled Python objects hit this.

## Shared data

Pre-generated training data, trained model weights, and MCMC results are in the Dropbox folder [NeuralOperator_Survey_Shared_Data_June2026](https://www.dropbox.com/scl/fo/co5v2bozvr5y8uv5kc29y/ACKiT1sBBQCTKV2wZYcAIlI?rlkey=agt87l1tf89g967gf8ofe5nik&st=3no4j03k&dl=0).

Each problem's `data/README.md` explains which subfolder to copy locally. The Bayesian application READMEs describe additional copies for trained surrogates and MCMC outputs.

## Citing this work

See `neural_operators`'s README for the article and code citations. The Zenodo-archived `survey26_v2` tag there is a complete snapshot, including this repo's `survey_work` as it stood before the split into two repos.
