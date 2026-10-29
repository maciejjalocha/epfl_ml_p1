"""Small NumPy preprocessor for the cleaning notebook; fit on training rows only.

Missingness filtering, imputation, scaling and category vocabularies are fitted
together. CSVs and labels are never modified. See cleaning_schema.json for the
field-specific codes: 7/8/9/77/88/99 cannot be cleaned with a global rule.
"""
import csv
from itertools import islice
import json
from pathlib import Path

import numpy as np

SCHEMA = json.loads(Path(__file__).with_name("cleaning_schema.json").read_text())["features"]


def load_cleaning_data(data_dir, chunk_size=5000, sub_sample=False):
    """Parse CSV rows in batches (5,000 by default) to reduce temporary memory use."""
    data_dir = Path(data_dir)
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")

    def read(filename):
        path = data_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Put the released dataset CSVs in {data_dir}: missing {filename}")
        with path.open() as f:
            header = next(csv.reader(f))
            count = sum(bool(line.strip()) for line in f)
        if not count or header[0] != "Id":
            raise ValueError(f"Expected nonempty Id-first CSV: {path}")
        ids = np.empty(count, dtype=np.int64)
        values = np.empty((count, len(header) - 1), dtype=np.float64)
        with path.open() as f:
            next(f)
            offset = 0
            lines = (line for line in f if line.strip())
            while batch := list(islice(lines, chunk_size)):
                block = np.genfromtxt(batch, delimiter=",", ndmin=2)
                if block.shape[1] != len(header) or not np.isfinite(block[:, 0]).all():
                    raise ValueError(f"Malformed CSV block in {path}")
                end = offset + len(block)
                ids[offset:end] = block[:, 0].astype(np.int64)
                values[offset:end] = block[:, 1:]
                offset = end
        return values, ids, header[1:]

    train, train_ids, names = read("x_train.csv")
    test, test_ids, test_names = read("x_test.csv")
    labels, label_ids, _ = read("y_train.csv")
    if names != test_names or labels.shape[1] != 1:
        raise ValueError("Train/test feature schemas or label shape disagree")
    if not np.array_equal(train_ids, label_ids):
        raise ValueError("Label IDs do not match training IDs")
    if not np.isin(labels, [-1, 1]).all():
        raise ValueError("Expected labels -1/+1")

    if sub_sample:
        # shuffle the data before sub-sampling to avoid biasing the sample
        rng = np.random.default_rng(seed=42)
        indices = rng.permutation(len(train))[:len(train) // 50]

        train = train[indices]
        train_ids = train_ids[indices]
        labels = labels[indices]
        # label_ids = label_ids[indices]

    return train, test, labels[:, 0].astype(int), train_ids, test_ids, names


def clean_column(values, spec):
    """Return cleaned values, unavailable mask, and interpretable numeric flags."""
    values = np.asarray(values, dtype=float).copy()
    blank = np.isnan(values)
    flags = {"blank": blank}
    unavailable = blank | np.isin(values, spec["sentinels"])
    if spec["kind"] == "numeric":
        for code in spec["sentinels"] + spec.get("non_numeric_codes", []):
            flags[f"response_{code}"] = values == code
        # Preserve special responses as flags, not numeric measurements.
        special = spec["sentinels"] + spec.get("non_numeric_codes", [])
        unavailable |= np.isin(values, special)
        values[np.isin(values, special)] = np.nan
        # Convert the zero-response codes listed in the schema to 0.
        values[np.isin(values, spec["zeros"])] = 0
    return values, unavailable, flags


class SurveyPreprocessor:
    """Compute means, scales and categories from training rows; reuse them unchanged."""

    def __init__(self, max_missing=0.5, manual_drop=None):
        if not 0 <= max_missing <= 1:
            raise ValueError("max_missing must be between 0 and 1")
        self.max_missing = max_missing
        self.manual_drop = dict(manual_drop or {})

    def fit(self, x, names):
        self.names_ = list(names)
        self._validate(x, names)
        if len(x) == 0:
            raise ValueError("Cannot fit on an empty dataset")
        self.columns_, self.dropped_, self.audit_ = [], {}, []
        self.feature_names_ = ["bias"]
        for j, name in enumerate(names):
            spec = SCHEMA[name]
            values, unavailable, flags = clean_column(x[:, j], spec)
            missing = float(unavailable.mean())
            reason = self.manual_drop.get(name)
            if reason is None and (missing > self.max_missing or missing == 1):
                reason = f"{missing:.1%} unavailable"
            self.audit_.append({"field": name, "kind": spec["kind"],
                                "missing_fraction": missing, "dropped": reason})
            if reason:
                self.dropped_[name] = reason
                continue
            if spec["kind"] == "numeric":
                mean = float(np.nanmean(values))
                filled = np.where(np.isnan(values), mean, values)
                scale = float(filled.std())
                if scale > 0:
                    self.columns_.append((j, "numeric", mean, scale))
                    self.feature_names_.append(name)
                for flag, v in flags.items():
                    if v.any() and not v.all():
                        self.columns_.append((j, "flag", flag, None))
                        self.feature_names_.append(f"{name}__{flag}")
            else:
                levels = np.unique(values[np.isfinite(values)]).tolist()
                # Missing is its own category, not a mean or a real state code.
                # Drop one reference category to avoid the dummy/intercept trap.
                if np.isnan(values).any():
                    levels.append(None)
                for level in levels[1:]:
                    self.columns_.append((j, "category", level, None))
                    label = "missing" if level is None else str(int(level))
                    self.feature_names_.append(f"{name}={label}")
        return self

    def transform(self, x, names):
        if not hasattr(self, "columns_"):
            raise ValueError("Call fit before transform")
        self._validate(x, names)
        out = np.ones((len(x), len(self.feature_names_)), dtype=np.float32)
        previous = None
        for k, (j, kind, arg, scale) in enumerate(self.columns_, 1):
            if j != previous:
                values, _, flags = clean_column(x[:, j], SCHEMA[self.names_[j]])
                previous = j
            if kind == "numeric":
                out[:, k] = (np.where(np.isnan(values), arg, values) - arg) / scale
            elif kind == "flag":
                out[:, k] = flags[arg]
            else:
                out[:, k] = np.isnan(values) if arg is None else values == arg
        # Categories unseen during fit yield all-zero dummies (reference level).
        # No validation/test category is allowed to change the output columns.
        if not np.isfinite(out).all():
            raise ValueError("Non-finite output; inspect input magnitudes")
        return out

    def _validate(self, x, names):
        if list(names) != self.names_ or len(set(names)) != len(names):
            raise ValueError("Feature names/order must match the fitted schema")
        if x.ndim != 2 or x.shape[1] != len(names):
            raise ValueError("Expected one name per input column")
        if set(names) - set(SCHEMA):
            raise ValueError("Unrecognized columns: update the explicit schema")
        if np.isinf(x).any():
            raise ValueError("Infinity is not a missing survey answer")
