"""
dgp.py

Single shared data-generating process and evaluation library for the
four-method comparison (DAGFR / Adaptive Lasso / Weighted Kernel / P-spline).

True structural sets (used for structure-recovery scoring across all methods):
  ZERO_TRUE = {cts1, cts4,...,cts15, cat0, cat1, cat2}   (16 true zero groups)
  TIED_TRUE = {cts3, func0}                                (2 tied groups)
  FREE_TRUE = {cts0, cts2}                                 (2 genuinely free)
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, log_loss, f1_score, precision_score, recall_score,
    confusion_matrix, roc_auc_score, cohen_kappa_score,
)

# ------------------------- global structural constants -------------------------

K = 3
KC = K - 1

N_CTS_SIGNAL = 4          # cts0..cts3 (cts1 true zero, cts0/cts2 free, cts3 tied)
N_CTS_NUISANCE = 12       # cts4..cts15, pure noise, coef=0
N_CTS_TOTAL = N_CTS_SIGNAL + N_CTS_NUISANCE   # 16

CAT_GROUP_NAMES = ['cat0', 'cat1', 'cat2']    # all three are true-zero groups
N_CAT_LEVELS = 3                                # each -> 2 dummy columns

N_FUNC_COEF = 3            # true Legendre expansion order for func0

CTS_CORR_RHO = 0.6          # equicorrelation across ALL 16 continuous predictors

T_GRID = 10
N_INTERVALS = 5
SIGMA_CURVE = 2.5
LAMBDA_GRID_SPLINE = np.logspace(-3, 4, 40)


def _build_groups():
    groups = []
    idx = 0
    for j in range(N_CTS_TOTAL):
        groups.append((f'cts{j}', slice(idx, idx + 1)))
        idx += 1
    for name in CAT_GROUP_NAMES:
        groups.append((name, slice(idx, idx + 2)))
        idx += 2
    groups.append(('func0', slice(idx, idx + N_FUNC_COEF)))
    idx += N_FUNC_COEF
    return groups, idx


GROUPS, P_TOTAL = _build_groups()
GROUP_NAMES = [name for name, _ in GROUPS]

ZERO_TRUE = ({'cts1'} | {f'cts{j}' for j in range(N_CTS_SIGNAL, N_CTS_TOTAL)}
             | set(CAT_GROUP_NAMES))
TIED_TRUE = {'cts3', 'func0'}
FREE_TRUE = {'cts0', 'cts2'}


def true_B():
    B = np.zeros((KC, P_TOTAL))
    idx = dict(GROUPS)
    B[0, idx['cts0']] = 1.5;  B[1, idx['cts0']] = -1.2
    B[0, idx['cts2']] = 1.0;  B[1, idx['cts2']] = -0.8
    B[0, idx['cts3']] = 0.9;  B[1, idx['cts3']] = 0.9
    B[0, idx['func0']] = [1.0, -0.8, 0.6]
    B[1, idx['func0']] = [1.0, -0.8, 0.6]
    return B


def softmax_probs(Xd, B):
    n_ = Xd.shape[0]
    logits = np.hstack([Xd @ B.T, np.zeros((n_, 1))])
    logits -= logits.max(axis=1, keepdims=True)
    ex = np.exp(logits)
    return ex / ex.sum(axis=1, keepdims=True)


def legendre_basis_3(t):
    p1 = t
    p2 = 0.5 * (3 * t ** 2 - 1)
    p3 = 0.5 * (5 * t ** 3 - 3 * t)
    return np.column_stack([p1, p2, p3])


def dummy_encode(levels, n_levels):
    n = len(levels)
    Z = np.zeros((n, n_levels - 1))
    for lv in range(1, n_levels):
        Z[levels == lv, lv - 1] = 1.0
    return Z


# ------------------------- data generation -------------------------

def generate_full(seed, n, T=T_GRID, sigma_curve=SIGMA_CURVE, rho=CTS_CORR_RHO):
    rng = np.random.RandomState(seed)

    # Equicorrelated continuous design (signal AND nuisance coordinates
    # alike): Sigma_jk = rho for j!=k, 1 for j=k. Cholesky factor applied to
    # iid standard normals preserves unit marginal variance per coordinate.
    Sigma = (1 - rho) * np.eye(N_CTS_TOTAL) + rho * np.ones((N_CTS_TOTAL, N_CTS_TOTAL))
    L_chol = np.linalg.cholesky(Sigma)
    Z = rng.randn(n, N_CTS_TOTAL)
    X_cts = Z @ L_chol.T

    cat_levels, cat_dummies = {}, {}
    for name in CAT_GROUP_NAMES:
        lv = rng.choice(N_CAT_LEVELS, size=n)
        cat_levels[name] = lv
        cat_dummies[name] = dummy_encode(lv, N_CAT_LEVELS)

    t_latent = rng.uniform(-1, 1, size=n)
    X_func0 = legendre_basis_3(t_latent)

    Xd = np.hstack([X_cts] + [cat_dummies[name] for name in CAT_GROUP_NAMES] + [X_func0])
    B_true = true_B()
    probs_true = softmax_probs(Xd, B_true)
    Ylab = np.array([rng.choice(K, p=probs_true[i]) for i in range(n)])

    t_grid = np.linspace(-1.0, 1.0, T)
    basis_grid = legendre_basis_3(t_grid)
    true_curve = X_func0 @ basis_grid.T
    noise = rng.randn(n, T) * sigma_curve
    Y_curve = true_curve + noise

    return {
        'Xd': Xd, 'Ylab': Ylab, 'B_true': B_true,
        'X_cts': X_cts, 'cat_levels': cat_levels, 'cat_dummies': cat_dummies,
        'X_func0': X_func0,
        'Y_curve': Y_curve, 't_grid': t_grid,
    }


def generate_train_test(seed, n_train, n_test, **kwargs):
    train = generate_full(seed, n_train, **kwargs)
    test = generate_full(seed + 10_000_000, n_test, **kwargs)
    return train, test


# ------------------------- prediction-quality evaluation (K-class) -------------------------

def expected_calibration_error(probs, y_true, y_pred, n_bins=10):
    confidences = probs.max(axis=1)
    accuracies = (y_pred == y_true).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    n = len(y_true)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (confidences > lo) & (confidences <= hi) if i > 0 else \
               (confidences >= lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        bin_conf = confidences[mask].mean()
        bin_acc = accuracies[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return ece


def evaluate_predictions(probs_test, Ylab_test, K_=K):
    pred = probs_test.argmax(axis=1)
    n = len(Ylab_test)
    onehot = np.zeros((n, K_))
    onehot[np.arange(n), Ylab_test] = 1.0
    labels = list(range(K_))

    acc = accuracy_score(Ylab_test, pred)
    ll = log_loss(Ylab_test, probs_test, labels=labels)
    brier = float(np.mean(np.sum((onehot - probs_test) ** 2, axis=1)))
    f1_macro = f1_score(Ylab_test, pred, average='macro', labels=labels, zero_division=0)
    f1_weighted = f1_score(Ylab_test, pred, average='weighted', labels=labels, zero_division=0)
    precision_macro = precision_score(Ylab_test, pred, average='macro', labels=labels, zero_division=0)
    recall_macro = recall_score(Ylab_test, pred, average='macro', labels=labels, zero_division=0)
    kappa = cohen_kappa_score(Ylab_test, pred)
    try:
        auc_ovr = roc_auc_score(Ylab_test, probs_test, multi_class='ovr', average='macro', labels=labels)
    except ValueError:
        auc_ovr = float('nan')
    cm = confusion_matrix(Ylab_test, pred, labels=labels)
    per_class_f1 = f1_score(Ylab_test, pred, average=None, labels=labels, zero_division=0)
    per_class_precision = precision_score(Ylab_test, pred, average=None, labels=labels, zero_division=0)
    per_class_recall = recall_score(Ylab_test, pred, average=None, labels=labels, zero_division=0)
    ece = expected_calibration_error(probs_test, Ylab_test, pred)

    return {
        'accuracy': acc, 'logloss': ll, 'brier': brier,
        'f1_macro': f1_macro, 'f1_weighted': f1_weighted,
        'precision_macro': precision_macro, 'recall_macro': recall_macro,
        'cohen_kappa': kappa, 'auc_ovr_macro': auc_ovr, 'ece': ece,
        'confusion_matrix': cm,
        'per_class_f1': per_class_f1, 'per_class_precision': per_class_precision,
        'per_class_recall': per_class_recall,
    }


# ------------------------- structure-recovery evaluation (all methods) -------------------------

def _prf_sets(hat_set, true_set):
    tp = len(hat_set & true_set)
    fp = len(hat_set - true_set)
    fn = len(true_set - hat_set)
    precision = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if len(true_set) == 0 else 0.0)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def evaluate_structure(zero_g_hat, tied_g_hat):
    all_names = set(GROUP_NAMES)
    free_g_hat = all_names - set(zero_g_hat) - set(tied_g_hat)
    zero_p, zero_r, zero_f1 = _prf_sets(set(zero_g_hat), ZERO_TRUE)
    tied_p, tied_r, tied_f1 = _prf_sets(set(tied_g_hat), TIED_TRUE)
    exact_match = (set(zero_g_hat) == ZERO_TRUE) and (set(tied_g_hat) == TIED_TRUE)
    return {
        'zero_precision': zero_p, 'zero_recall': zero_r, 'zero_f1': zero_f1,
        'tied_precision': tied_p, 'tied_recall': tied_r, 'tied_f1': tied_f1,
        'exact_structure_match': exact_match,
        'zero_g_hat': set(zero_g_hat), 'tied_g_hat': set(tied_g_hat), 'free_g_hat': free_g_hat,
    }


def derive_zero_set_from_magnitudes(magnitudes: dict, rel_thresh=0.05):
    vals = list(magnitudes.values())
    max_mag = max(vals) if len(vals) > 0 and max(vals) > 0 else 1.0
    return {name for name, val in magnitudes.items() if val <= rel_thresh * max_mag}


# ------------------------- coefficient-recovery evaluation (DAGFR / Lasso only) -------------------------

def evaluate_coefficients(B_hat, B_true):
    diff = B_hat - B_true
    fro_error = float(np.linalg.norm(diff))
    fro_true = float(np.linalg.norm(B_true))
    rel_fro_error = fro_error / fro_true if fro_true > 0 else float('nan')
    max_abs_error = float(np.max(np.abs(diff)))
    per_group = {}
    for name, sl in GROUPS:
        per_group[name] = {
            'true_norm': float(np.linalg.norm(B_true[:, sl])),
            'hat_norm': float(np.linalg.norm(B_hat[:, sl])),
            'error_norm': float(np.linalg.norm(diff[:, sl])),
        }
    return {
        'frobenius_error': fro_error, 'relative_frobenius_error': rel_fro_error,
        'max_abs_error': max_abs_error, 'per_group': per_group,
    }