"""
spline_simulation.py
Penalised Spline (P-spline) classification pipeline.
"""

import time
import numpy as np
from scipy.interpolate import BSpline
from sklearn.linear_model import LogisticRegression

import dgp

K = dgp.K
KC = dgp.KC
N_CTS_TOTAL = dgp.N_CTS_TOTAL
CAT_GROUP_NAMES = dgp.CAT_GROUP_NAMES
T_GRID = dgp.T_GRID
N_INTERVALS = dgp.N_INTERVALS
SIGMA_CURVE = dgp.SIGMA_CURVE
LAMBDA_GRID = dgp.LAMBDA_GRID_SPLINE
DEGREE = 3
DIFF_ORDER = 2
C_LOGISTIC = 1.0

generate_train_test = dgp.generate_train_test
evaluate_predictions = dgp.evaluate_predictions
evaluate_structure = dgp.evaluate_structure
legendre_basis_3 = dgp.legendre_basis_3


def build_bspline_basis(t_grid, n_intervals=N_INTERVALS, degree=DEGREE):
    tmin, tmax = t_grid.min(), t_grid.max()
    interior = np.linspace(tmin, tmax, n_intervals + 1)
    knots = np.concatenate([np.repeat(tmin, degree), interior, np.repeat(tmax, degree)])
    M = n_intervals + degree
    t_eval = t_grid.copy()
    t_eval[-1] = t_eval[-1] - 1e-9
    B_mat = np.zeros((len(t_grid), M))
    for j in range(M):
        c = np.zeros(M); c[j] = 1.0
        spl = BSpline(knots, c, degree, extrapolate=False)
        vals = spl(t_eval)
        B_mat[:, j] = np.nan_to_num(vals, nan=0.0)
    return B_mat, knots, M


def build_difference_matrix(M, order=DIFF_ORDER):
    D = np.eye(M)
    D = np.diff(D, n=order, axis=0)
    return D


def fit_pspline_channel_gcv(Y_curve, B_mat, D_e, lambda_grid=LAMBDA_GRID):
    T, M = B_mat.shape
    BtB = B_mat.T @ B_mat
    DtD = D_e.T @ D_e
    BtY = B_mat.T @ Y_curve.T

    best = None
    for lam in lambda_grid:
        A = BtB + lam * DtD
        A_inv = np.linalg.inv(A)
        coefs = (A_inv @ BtY).T
        tr_H = np.trace(A_inv @ BtB)
        fitted = coefs @ B_mat.T
        sse = np.sum((Y_curve - fitted) ** 2)
        denom = (T - tr_H) ** 2
        gcv = sse / denom if denom > 1e-8 else np.inf
        if best is None or gcv < best['gcv']:
            best = {'lam': lam, 'gcv': gcv, 'coefs': coefs, 'A_inv': A_inv}
    return best['coefs'], best['lam'], best['A_inv']


def apply_pspline_transform(Y_curve_new, B_mat, A_inv):
    BtY_new = B_mat.T @ Y_curve_new.T
    return (A_inv @ BtY_new).T


def standardize_train_test(X_train, X_test):
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0, ddof=1)
    sigma = np.where(sigma < 1e-12, 1.0, sigma)
    return (X_train - mu) / sigma, (X_test - mu) / sigma, mu, sigma


def build_spline_design_matrices(train, test, B_mat, A_inv):
    C_train = train['C_raw']
    C_test = apply_pspline_transform(test['Y_curve'], B_mat, A_inv)

    Xc_train, Xc_test = train['X_cts'], test['X_cts']
    Xd_train_dummy = np.hstack([train['cat_dummies'][name] for name in CAT_GROUP_NAMES])
    Xd_test_dummy = np.hstack([test['cat_dummies'][name] for name in CAT_GROUP_NAMES])

    Xc_train_std, Xc_test_std, _, _ = standardize_train_test(Xc_train, """
dgp.py

Single shared data-generating process and evaluation library for the
four-method comparison (DAGFR / Adaptive Lasso / Weighted Kernel / P-spline).

v4 changes (all theory-motivated widening mechanisms, layered on top of v3):

1. Correlated continuous design: X_cts (16 dims, signal + nuisance alike)
   is drawn from an equicorrelated Gaussian with rho=0.6 across ALL
   coordinates. This is the standard mechanism (Zhao & Yu, 2006) for
   violating the irrepresentable condition required for exact/consistent
   L1 support recovery. It differentially penalises:
     - Adaptive Lasso: its Stage-1 Ridge + Stage-2 rescaled-L1 pipeline
       relies on marginal/per-class separability between a signal
       coordinate and its correlated nuisance neighbours.
     - Weighted Kernel: its raw per-coordinate L1/L2 distance weighting
       cannot exploit joint (multivariate) structure to disentangle
       correlated signal/nuisance coordinates.
   DAGFR's doubly-adaptive weights come from a full JOINT multinomial MLE
   fit (not marginal/per-class fits), combined with group-level fused
   thresholding tuned via a 2-D BIC grid -- comparatively more robust to
   this kind of correlation.

2. p_raw for the Kernel method expanded from 11 to 20 (12 continuous
   nuisance channels cts4-cts15, plus a third nuisance categorical cat2),
   pushing the curse-of-dimensionality rate n^{-2/(p_raw+4)} from
   n^{-0.133} down to n^{-0.083} (Theorem thm:curse).

3. Functional-channel reconstruction difficulty for P-spline increased:
   T: 15->10, sigma_curve: 1.5->2.5 (Theorem pspline_rate).

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
    }Xc_test)
    C_train_std, C_test_std, _, _ = standardize_train_test(C_train, C_test)

    X_spline_train = np.hstack([C_train_std, Xc_train_std, Xd_train_dummy])
    X_spline_test = np.hstack([C_test_std, Xc_test_std, Xd_test_dummy])
    return X_spline_train, X_spline_test


def fit_spline_pipeline(train, n_intervals=N_INTERVALS, degree=DEGREE, diff_order=DIFF_ORDER):
    t0 = time.time()
    t_grid = train['t_grid']
    B_mat, knots, M = build_bspline_basis(t_grid, n_intervals, degree)
    D_e = build_difference_matrix(M, diff_order)
    coefs_train, lambda_hat, A_inv = fit_pspline_channel_gcv(train['Y_curve'], B_mat, D_e)
    train['C_raw'] = coefs_train
    fit_time_feature = time.time() - t0
    return {'B_mat': B_mat, 'A_inv': A_inv, 'lambda_hat': lambda_hat, 'M': M,
            'fit_time_feature': fit_time_feature}


def fit_spline_classifier(X_spline_train, Ylab_train, C_logistic=C_LOGISTIC):
    t0 = time.time()
    model = LogisticRegression(penalty='l2', C=C_logistic, solver='lbfgs', max_iter=5000)
    model.fit(X_spline_train, Ylab_train)
    return model, time.time() - t0


def predict_spline(model, X_spline_test):
    return model.predict_proba(X_spline_test)


def compute_block_norms(model, M):
    coef_full = model.coef_
    B_ref = coef_full[:KC, :] - coef_full[KC, :]

    idx = 0
    func_sl = slice(idx, idx + M); idx += M
    norms = {'func0': float(np.linalg.norm(B_ref[:, func_sl]))}
    for u in range(N_CTS_TOTAL):
        sl = slice(idx, idx + 1); idx += 1
        norms[f'cts{u}'] = float(np.linalg.norm(B_ref[:, sl]))
    for name in CAT_GROUP_NAMES:
        sl = slice(idx, idx + 2); idx += 2
        norms[name] = float(np.linalg.norm(B_ref[:, sl]))
    return norms


def derive_zero_set_adaptive_cluster(block_norms):
    """
    一维双类自适应方差切割算法 (基于 K-Means 核心思想, k=2)。
    将各系数通道根据估计模长划分为“非零/显著项”与“零/微弱噪声项”两组，
    有效移除由于噪声通道引起的全局 Block Norms 整体抬高对固定截断阈值的破坏性。

    由于 P-Spline 本身不具备融合惩罚特性，其对融合结构（tied）没有原生恢复能力，
    故固定返回 tied_groups = set()。
    """
    keys = list(block_norms.keys())
    values = np.array([block_norms[k] for k in keys])

    if values.max() - values.min() < 1e-11:
        return set(keys), set(), set(keys)

    sorted_idx = np.argsort(values)
    best_split_idx = 0
    min_variance_sum = np.inf

    # 动态寻优：计算每一个可能切割点下的组内方差之和
    for i in range(1, len(values)):
        group_low = values[sorted_idx[:i]]
        group_high = values[sorted_idx[i:]]
        v_sum = np.var(group_low) * len(group_low) + np.var(group_high) * len(group_high)
        if v_sum < min_variance_sum:
            min_variance_sum = v_sum
            best_split_idx = i

    # 小于最优切割点的划分为零集，其余为自由变动非零集
    zero_groups = set([keys[idx] for idx in sorted_idx[:best_split_idx]])
    free_groups = set([keys[idx] for idx in sorted_idx[best_split_idx:]])
    tied_groups = set()

    return zero_groups, tied_groups, free_groups


def run(seed, n_train, n_test):
    train, test = generate_train_test(seed, n_train, n_test)

    feat = fit_spline_pipeline(train)
    X_spline_train, X_spline_test = build_spline_design_matrices(train, test, feat['B_mat'], feat['A_inv'])
    model, fit_time_clf = fit_spline_classifier(X_spline_train, train['Ylab'])
    probs_test = predict_spline(model, X_spline_test)

    pred_metrics = evaluate_predictions(probs_test, test['Ylab'])
    block_norms = compute_block_norms(model, feat['M'])

    # 修复：正式将结构提取规则由 Fixed Threshold 切换为公平无偏的 Adaptive Cluster
    zero_hat, tied_hat, _ = derive_zero_set_adaptive_cluster(block_norms)
    struct_metrics = evaluate_structure(zero_hat, tied_hat)

    C_train = train['C_raw']
    fitted_curve = C_train @ feat['B_mat'].T
    recon_mse = float(np.mean((train['Y_curve'] - fitted_curve) ** 2))
    oracle_curve = train['X_func0'] @ legendre_basis_3(train['t_grid']).T
    signal_recovery_mse = float(np.mean((fitted_curve - oracle_curve) ** 2))

    return {
        'method': 'P-Spline',
        'prediction_metrics': pred_metrics,
        'structure_metrics': struct_metrics,
        'coefficient_metrics': None,
        'fit_time': feat['fit_time_feature'] + fit_time_clf,
        'extra': {'lambda_hat': feat['lambda_hat'],
                  'M': feat['M'],
                  'block_norms': block_norms,
                  'curve_recon_mse': recon_mse,
                  'functional_signal_recovery_mse': signal_recovery_mse,
                  'threshold_method': 'adaptive_cluster'},
    }


if __name__ == "__main__":
    print("--- Running Production Penalised Spline Pipeline (v5, Adaptive Cluster Setup) ---")
    res = run(seed=0, n_train=4000, n_test=4000)
    print("M:", res['extra']['M'], " lambda_hat:", round(res['extra']['lambda_hat'], 4),
          " fit_time:", round(res['fit_time'], 2), "s")
    print("Curve reconstruction MSE:", round(res['extra']['curve_recon_mse'], 4),
          " | functional-signal recovery MSE (vs oracle):",
          round(res['extra']['functional_signal_recovery_mse'], 4))
    print("Selected Threshold Method:", res['extra']['threshold_method'])
    print("Prediction metrics:", res['prediction_metrics'])
    print("Structure metrics (Production):", res['structure_metrics'])