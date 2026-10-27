"""Disk caches for raw data, cleaned matrices and fitted models."""
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np

from p1.cleaning import SurveyPreprocessor, load_cleaning_data
from p1.sparse_numpy import SparseMatrix
from p1.training import tune_logistic


def cache_key(data_dir, settings=None):
    files = [Path(data_dir) / n for n in ('x_train.csv', 'x_test.csv', 'y_train.csv')]
    # Size/mtime detect ordinary CSV edits without rereading a gigabyte each run.
    stamp = [(str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns) for p in files]
    code = b''.join(Path(__file__).with_name(n).read_bytes()
                    for n in ('cleaning.py', 'cleaning_schema.json', 'sparse_numpy.py'))
    return hashlib.sha256(code + json.dumps([stamp, settings], sort_keys=True).encode()).hexdigest()


def cached_raw(data_dir, cache_dir):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / 'raw.pkl'
    key = cache_key(data_dir)
    if path.exists():
        with path.open('rb') as f:
            if pickle.load(f) == key:
                print('Raw data: cache hit')
                return pickle.load(f)
    data = load_cleaning_data(data_dir)
    with path.open('wb') as f:
        pickle.dump(key, f)
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print('Raw data: cached')
    return data


def cached_clean(x, test, names, splits, data_dir, cache_dir, max_missing=.5, manual_drop=None):
    cache_dir = Path(cache_dir)
    key = cache_key(data_dir, {'max_missing': max_missing, 'manual_drop': manual_drop,
        'splits': [hashlib.sha256(i.tobytes()).hexdigest() for i in splits]})
    meta = cache_dir / 'clean.pkl'
    paths = [cache_dir / f'{name}.pkl' for name in ('fit', 'tune', 'valid', 'test')]
    if meta.exists() and all(p.exists() for p in paths):
        with meta.open('rb') as f:
            saved_key, prep = pickle.load(f)
        if key == saved_key:
            print('Cleaned matrices: cache hit')
            return prep, *(_read_pickle(p) for p in paths)
    # Write metadata last: an interrupted build must not be accepted as complete.
    meta.unlink(missing_ok=True)
    prep = SurveyPreprocessor(max_missing=max_missing, manual_drop=manual_drop).fit(x[splits[0]], names)
    for p, indices in zip(paths, (*splits, None)):
        source = test if indices is None else x
        indices = np.arange(len(test)) if indices is None else indices
        blocks = (prep.transform(source[indices[i:i+4096]], names)
                  for i in range(0, len(indices), 4096))
        matrix = SparseMatrix.from_blocks(blocks, len(prep.feature_names_))
        with p.open('wb') as f:
            pickle.dump(matrix, f, protocol=pickle.HIGHEST_PROTOCOL)
        del matrix
    with meta.open('wb') as f:
        pickle.dump((key, prep), f, protocol=pickle.HIGHEST_PROTOCOL)
    print('Cleaned matrices: cached')
    return prep, *(_read_pickle(p) for p in paths)


def cached_tuning(x, y, tune_x, tune_y, cache_dir, *, feature_names=None, **settings):
    cache_dir = Path(cache_dir)
    key = hashlib.sha256((cache_dir / 'clean.pkl').read_bytes() +
                        Path(__file__).read_bytes() + Path(__file__).with_name('training.py').read_bytes() +
                        (Path(__file__).parent / 'modeling/train.py').read_bytes() + json.dumps([settings, feature_names], sort_keys=True).encode() +
                        np.asarray(y).tobytes() + np.asarray(tune_y).tobytes()).hexdigest()
    path = cache_dir / 'logistic_gd.pkl'
    if path.exists():
        with path.open('rb') as f:
            saved_key, result = pickle.load(f)
        if key == saved_key:
            print('Logistic search: cache hit')
            return result
    result = tune_logistic(x, y, tune_x, tune_y, **settings)
    with path.open('wb') as f:
        pickle.dump((key, result), f, protocol=pickle.HIGHEST_PROTOCOL)
    return result


def _read_pickle(path):
    with path.open('rb') as f:
        return pickle.load(f)
