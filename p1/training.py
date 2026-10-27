"""NumPy logistic training, stratified splitting and F1 threshold selection."""
import numpy as np

from p1.modeling import train


def split_rows(y, seed=2026):
    """80/10/10 stratified split; fitting, tuning, evaluation."""
    rng = np.random.default_rng(seed)
    groups = [rng.permutation(np.flatnonzero(y == label)) for label in (0, 1)]
    parts = [np.split(g, [int(.8*len(g)), int(.9*len(g))]) for g in groups]
    return tuple(np.concatenate(part) for part in zip(*parts))



def best_threshold(y, scores):
    """Exact F1 sweep; evaluate tied scores together (predict score >= cutoff)."""
    order = np.argsort(-scores, kind='stable')
    sorted_scores = scores[order]
    ends = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(y)-1]
    tp = np.cumsum(y[order])[ends]
    f1 = 2 * tp / (ends + 1 + y.sum())
    best = np.argmax(f1)
    return float(sorted_scores[ends[best]]), float(f1[best])


def tune_logistic(x, y, tune_x, tune_y, rates=(.01, .03, .1), penalties=(1e-4, 1e-3),
                  max_steps=500, patience=40, min_delta=1e-5):
    """Tune regularized logistic regression one step at a time; restore the best tuning checkpoint."""
    y = np.asarray(y, dtype=np.float32)
    trials = []
    for penalty in penalties:
        for rate in rates:
            w = np.zeros(x.shape[1], dtype=np.float32)
            best_loss, stale, history = np.inf, 0, []
            for step in range(1, max_steps + 1):
                w, _ = train.reg_logistic_regression(y, x, penalty, w, 1, rate)
                loss = train.nll(tune_y, tune_x, w)
                if not np.isfinite(loss):
                    raise ValueError('Non-finite logistic loss')
                history.append(loss)
                if loss < best_loss - min_delta:
                    best_loss, best_w, best_step, stale = loss, w.copy(), step, 0
                else:
                    stale += 1
                if stale >= patience:
                    break
            threshold, score = best_threshold(tune_y, train.sigmoid(tune_x @ best_w))
            trial = dict(rate=rate, penalty=penalty, weights=best_w, step=best_step,
                         stopped_early=stale >= patience, loss=best_loss,
                         history=history, threshold=threshold, f1=score)
            trials.append(trial)
            print(f'lr={rate:g}, lambda={penalty:g}: tune F1={score:.4f}, '
                  f'BCE={best_loss:.4f}, best step={best_step}/{step}', flush=True)
    selected = max(trials, key=lambda t: (t['f1'], -t['loss']))
    return selected, trials
