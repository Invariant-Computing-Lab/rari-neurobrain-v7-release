# RARI NeuroBrain V7 offline release candidate

CP05. G01, G03–G09 retain their catalog identities within one source lineage, not eight new studies. No cloud/provider call occurs in the supported entry points.

## Install and run
```sh
python -m pip install "numpy==2.3.5"
python -m pip install --no-index --no-deps wheels/rari_neurobrain_v7_release-0.1.0-py3-none-any.whl
python -m rari_neurobrain.release assets/MODELS.json assets/example_features.json
python -m unittest discover -s tests -v
```
The tested release pins NumPy 2.3.5. The fixture is one already-seen development record, not a fresh test. Weights remain exactly the preserved V7 `MODELS.json`; 522 parameter sets are 3 maps × 29 widths × 6 tasks, not 522 independent AI models. V7 selects PCA/12. The prior OMEGA model chooses PCA/11 and is a different fitted bank.

## Input/output contract
Features have shape [batch,1212]: flattened mean, population-standard-deviation and last value of the source eight-tick [8,101,4] pre-perturbation observations. Raw EEG/MRI, arbitrary time series and text are not accepted. Task dimensions are 2,4,16,1,2,2 in the source task order. `fit_observer(u,y)` requires finite [n,2] known intervention/response arrays with n>=6; linear/tanh model assumptions and known sensor inversion apply. Fitting a rank-deficient map returns an observable representative, not missing latent coordinates.

## Provenance and limitations
`reference_source` is preserved source for inspection, not a claim that all runtime infrastructure was retested. Original algorithms in src retain their hashes. `release.py` is a separately named validation wrapper; it does not change scientific estimators. The standalone package excludes seed-vault and provider credentials. Source code, archive outputs and tests do not establish biological validation, general observation recovery, safety or clinical efficacy.

## Rights
No license is newly granted. Author confirmation of rights and public release is required. The original LICENSE_NOTICE is preserved.

## Tested numerical environment
CP05 release environment is Python 3.13.5 with NumPy 2.3.5. A separate NumPy 2.2.4 run passed 16/17 tests but failed the strict 1e-12 saved-fixture comparison at an observed 3.60e-12 difference. The tolerance was not relaxed and the failure is preserved; NumPy 2.2.4 is not certified by this release. The clean wheel environment shares the explicitly identified preinstalled NumPy 2.3.5 distribution, rather than claiming all dependencies were downloaded or rebuilt offline.
## Verified packaging revision — 2026-10-20



Runtime state: `PASS`. Rights and release approval are separate. See `RELEASE_VERIFICATION.json`. No paper or conference submission is included as a new software result.
