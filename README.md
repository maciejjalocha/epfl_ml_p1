# EPFL Machine Learning — Project 1

This repository contains our Project 1 work for the EPFL Machine Learning
course. The project is a binary prediction task on Behavioral Risk Factor
Surveillance System (BRFSS) data, with coronary heart disease as the target.

## What happened in this repository

The work was developed in stages:

1. The original CSV data was explored and cleaned in the notebooks under
   [`notebooks/`](./notebooks).
2. The cleaning pipeline was moved into reusable Python code under
   [`p1/`](./p1). It handles:
   - parsing the training and test CSV files;
   - field-specific missing-value and response-code handling;
   - fitting preprocessing only on the fitting split;
   - numeric imputation and scaling;
   - categorical one-hot encoding;
   - sparse feature storage to avoid materialising unnecessarily large dense
     matrices.
3. Several models were implemented and compared, including least squares,
   ridge regression, linear GD/SGD, logistic regression, and regularised
   logistic regression.
4. The final experiments use a deterministic stratified split, validation
   tuning, early stopping, feature filtering, and threshold selection for F1.
5. Figures generated during the experiments are stored in
   [`reports/figures/`](./reports/figures).

The notebooks are the experiment record. The Python package contains the
reusable implementation of the data preparation and modelling code.

## Required course implementation

The six functions required by the Project 1 specification are implemented in
[`p1/modeling/implementations.py`](./p1/modeling/implementations.py). This
file is a copy of the modelling implementation previously developed in
[`p1/modeling/train.py`](./p1/modeling/train.py), with the required function
interfaces:

- `mean_squared_error_gd`
- `mean_squared_error_sgd`
- `least_squares`
- `ridge_regression`
- `logistic_regression`
- `reg_logistic_regression`

The implementations use NumPy only for the machine-learning operations. The
returned loss for ridge and regularised logistic regression is the unpenalised
loss, as required by the project specification.

## Reproducing the experiments

The project uses `uv` for the Python environment. With Python 3.12
available, install the locked dependencies with:

```bash
uv sync
```

In order to have the data prepared, insert the dataset zip file in `/data/raw/` folder.

Then, run `make data`.

The cached intermediate data in `data/interim/` is generated from these
inputs. Do not rely on the large raw dataset or generated caches being
committed to GitHub.


### Data

Head into `notebokks/data-cleaning-correlated.ipynb` and run the code.

### Experiments

Head into `notebooks/compare_models.ipynb`

### Others
In the `notebooks` you can see more notebooks containing extra plots and extra experiments.


## Repository layout

```text
.
├── p1/                  reusable data, feature, and modelling code
├── p1/modeling/         model implementations and prediction helpers
├── notebooks/           exploration, cleaning, and model comparison
├── reports/figures/     figures used in the report
├── models/              saved model artefacts
├── data/                local data and generated caches (not for submission)
├── pyproject.toml       project metadata and dependencies
├── uv.lock              locked dependency versions
└── README.md            this document
```
