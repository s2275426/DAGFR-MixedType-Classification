"""
Adaptive Lasso baseline (OvR, Ridge-initialized, BIC-selected).
"""

import time
import numpy as np
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')

import dgp

K = dgp.K
KC = dgp.KC
P_TOTAL = dgp.P_TOTAL
GROUPS = dgp.GROUPS
generate_train_test = dgp.generate_train_test
evaluate_predictions = dgp.evaluate_predictions
evaluate_structure = dgp.evaluate_structure
evaluate_coefficients = dgp.evaluate_coefficients

GAMMA_ALASSO = 1.0
V_FOLDS = 5
TAU_FALLBACK = 0.5
EPS_WEIGHT_FLOOR = 1e-4


def fit_ridge_stage1(X, y, V=V_FOLDS, seed=0):
    Cs = np.logspace(-3, 3, 30)
    cv = KFold(n_splits=V, shuffle=True, random_state=seed)
    model = LogisticRegressionCV(Cs=Cs, cv=cv, penalty='l2', solver='lbfgs',
                                 max_iter=5000, scoring='neg_log_loss')
    model.fit(X, y)
    return model.coef_.ravel()


def compute_adaptive_weights(beta_ridge, gamma=GAMMA_ALASSO, eps=EPS_WEIGHT_FLOOR):
    return 1.0 / (np.abs(beta_ridge) + eps) ** gamma


def fit_weighted_lasso_logistic(X, y, weights, lam):
    X_scaled = X / weights[np.newaxis, :]
    C = 1.0 / lam if lam > 0 else 1e10
    model = LogisticRegression(penalty='l1', solver='liblinear', C=C,
                               max_iter=5000, fit_intercept=True)
    model.fit(X_scaled, y)
    alpha = model.coef_.ravel()
    beta = alpha / weights
    intercept = model.intercept_[0]
    return beta, intercept


def bic_binary(X, y, beta, intercept, df):
    n = X.shape[0]
    logits = X @ beta + intercept"""
kernel_simulation.py

Weighted Kernel baseline (Selk 2023), per Section 3.3.
Now imports the shared dgp.py module. Raw predictor count expanded to
p_raw=20 (16 continuous + 3 categorical + 1 functional) to scale up
the curse-of-dimensionality analysis. Casts distance tensors to float32
to manage memory footprint at n=4000.
"""

import time
import numpy as np
from scipy.optimize import minimize, minimize_scalar

import dgp

K = dgp.K
KC = dgp.KC
N_CTS_TOTAL = dgp.N_CTS_TOTAL          # 16
CAT_GROUP_NAMES = dgp.CAT_GROUP_NAMES  # ['cat0','cat1','cat2']

P_RAW_NAMES = [f'cts{u}' for u in range(N_CTS_TOTAL)] + CAT_GROUP_NAMES + ['func0']
P_RAW = len(P_RAW_NAMES)  # 16 + 3 + 1 = 20

OMEGA_UPPER = 10.0
LAMBDA_RIDGE = 0.05
N_TRAIN_KERNEL = 4000
N_TEST_KERNEL = 4000

generate_train_test = dgp.generate_train_test
evaluate_predictions = dgp.evaluate_predictions
evaluate_structure = dgp.evaluate_structure
derive_zero_set_from_magnitudes = dgp.derive_zero_set_from_magnitudes


def build_raw_distance_tensors(raw_a, raw_b, norm_consts):
    """
    NOTE: results are cast to float32 after normalisation to control memory
    at p_raw=20 (20 * n x n float64 tensors would be ~2.5GB at n=4000;
    float32 halves this to ~1.3GB without materially affecting the LOO
    Brier optimisation, which only needs modest numerical precision).
    """
    d = {}
    X_cts_a, X_cts_b = raw_a['X_cts'], raw_b['X_cts']
    for u in range(N_CTS_TOTAL):
        raw_dist = np.abs(X_cts_a[:, u:u+1] - X_cts_b[:, u].reshape(1, -1))
        d[f'cts{u}'] = (raw_dist / norm_consts[f'cts{u}']).astype(np.float32)

    for cat_name in CAT_GROUP_NAMES:
        cat_a, cat_b = raw_a['cat_levels'][cat_name], raw_b['cat_levels'][cat_name]
        raw_dist_cat = (cat_a.reshape(-1, 1) != cat_b.reshape(1, -1)).astype(np.float32)
        d[cat_name] = raw_dist_cat / norm_consts[cat_name]

    Xf_a, Xf_b = raw_a['X_func0'], raw_b['X_func0']
    diff = Xf_a[:, np.newaxis, :] - Xf_b[np.newaxis, :, :]
    raw_dist_fun = np.sqrt(np.sum(diff ** 2, axis=2))
    d['func0'] = (raw_dist_fun / norm_consts['func0']).astype(np.float32)
    return d


def compute_norm_constants(raw_train):
    n = raw_train['X_cts'].shape[0]
    consts = {}
    for u in range(N_CTS_TOTAL):
        consts[f'cts{u}'] = np.std(raw_train['X_cts'][:, u], ddof=1)

    for cat_name in CAT_GROUP_NAMES:
        z = (raw_train['cat_levels'][cat_name] == 0).astype(float)
        c = np.std(z, ddof=1)
        consts[cat_name] = c if c > 0 else 1e-8

    Xf = raw_train['X_func0']
    mean_curve = Xf.mean(axis=0)
    sq = np.sum((Xf - mean_curve) ** 2, axis=1)
    consts['func0'] = np.sqrt(np.sum(sq) / (n - 1))
    return consts


def picard_kernel(D):
    return np.exp(-D)


def combine_distance(dist_tensors, omega):
    D = np.zeros_like(dist_tensors[P_RAW_NAMES[0]])
    for j, name in enumerate(P_RAW_NAMES):
        D += omega[j] * dist_tensors[name]
    return D


def loo_posterior(dist_tensors_train, Y_train, omega, K_classes):
    n = len(Y_train)
    D = combine_distance(dist_tensors_train, omega)
    Kmat = picard_kernel(D)
    np.fill_diagonal(Kmat, 0.0)
    onehot = np.zeros((n, K_classes))
    onehot[np.arange(n), Y_train] = 1.0
    numer = Kmat @ onehot
    denom = Kmat.sum(axis=1, keepdims=True)
    denom = np.where(denom < 1e-300, 1e-300, denom)
    return numer / denom, onehot


def brier_loocv(dist_tensors_train, Y_train, omega, K_classes):
    P_loo, onehot = loo_posterior(dist_tensors_train, Y_train, omega, K_classes)
    return np.sum((onehot - P_loo) ** 2)


def objective_ridge(omega, dist_tensors_train, Y_train, K_classes, lam):
    Q = brier_loocv(dist_tensors_train, Y_train, omega, K_classes)
    return Q + lam * np.sum(omega ** 2)


def single_predictor_warm_start(dist_tensors_train, Y_train, K_classes, omega_upper):
    omega0 = np.zeros(P_RAW)
    for j, name in enumerate(P_RAW_NAMES):
        def q1(w, name=name):
            n = len(Y_train)
            D = w * dist_tensors_train[name]
            Kmat = picard_kernel(D)
            np.fill_diagonal(Kmat, 0.0)
            onehot = np.zeros((n, K_classes))
            onehot[np.arange(n), Y_train] = 1.0
            numer = Kmat @ onehot
            denom = Kmat.sum(axis=1, keepdims=True)
            denom = np.where(denom < 1e-300, 1e-300, denom)
            P_loo = numer / denom
            return np.sum((onehot - P_loo) ** 2)
        res = minimize_scalar(q1, bounds=(0.0, omega_upper), method='bounded',
                              options={'xatol': 1e-2})
        omega0[j] = res.x
    return omega0


def fit_kernel_weights(raw_train, Y_train, K_classes,
                       omega_upper=OMEGA_UPPER, lam=LAMBDA_RIDGE):
    t0 = time.time()
    norm_consts = compute_norm_constants(raw_train)
    dist_tensors_train = build_raw_distance_tensors(raw_train, raw_train, norm_consts)
    omega0 = single_predictor_warm_start(dist_tensors_train, Y_train, K_classes, omega_upper)

    bounds = [(0.0, omega_upper)] * P_RAW
    res = minimize(objective_ridge, omega0,
                   args=(dist_tensors_train, Y_train, K_classes, lam),
                   method='L-BFGS-B', bounds=bounds,
                   options={'maxiter': 200, 'eps': 1e-3})
    omega_hat = res.x
    fit_time = time.time() - t0
    return {'omega_hat': omega_hat, 'omega0': omega0, 'norm_consts': norm_consts,
            'fit_time': fit_time, 'converged': res.success}


def predict_kernel(model, raw_train, Y_train, raw_test, K_classes):
    dist_tensors_test = build_raw_distance_tensors(raw_test, raw_train, model['norm_consts'])
    D = combine_distance(dist_tensors_test, model['omega_hat'])
    Kmat = picard_kernel(D)
    onehot_train = np.zeros((len(Y_train), K_classes))
    onehot_train[np.arange(len(Y_train)), Y_train] = 1.0
    numer = Kmat @ onehot_train
    denom = Kmat.sum(axis=1, keepdims=True)
    denom = np.where(denom < 1e-300, 1e-300, denom)
    return numer / denom


def run(seed, n_train, n_test):
    train, test = generate_train_test(seed, n_train, n_test)
    raw_tr = {'X_cts': train['X_cts'], 'cat_levels': train['cat_levels'], 'X_func0': train['X_func0']}
    raw_te = {'X_cts': test['X_cts'], 'cat_levels': test['cat_levels'], 'X_func0': test['X_func0']}

    model = fit_kernel_weights(raw_tr, train['Ylab'], K)
    probs_te = predict_kernel(model, raw_tr, train['Ylab'], raw_te, K)

    pred_metrics = evaluate_predictions(probs_te, test['Ylab'])
    omega_dict = dict(zip(P_RAW_NAMES, model['omega_hat']))
    zero_hat = derive_zero_set_from_magnitudes(omega_dict)
    struct_metrics = evaluate_structure(zero_hat, set())

    return {
        'method': 'Weighted Kernel',
        'prediction_metrics': pred_metrics,
        'structure_metrics': struct_metrics,
        'coefficient_metrics': None,
        'fit_time': model['fit_time'],
        'extra': {'omega_hat': omega_dict, 'converged': model['converged'], 'p_raw': P_RAW},
    }


if __name__ == "__main__":
    print("--- Running Weighted Kernel Pipeline (v4, p_raw=20, shared dgp.py) ---")
    res = run(seed=0, n_train=N_TRAIN_KERNEL, n_test=N_TEST_KERNEL)
    print("omega_hat:", {k: round(float(v), 3) for k, v in res['extra']['omega_hat'].items()})
    print("converged:", res['extra']['converged'], " fit_time:", round(res['fit_time'], 2), "s")
    print("Prediction metrics:", res['prediction_metrics'])
    print("Structure metrics:", res['structure_metrics'])
    p = 1.0 / (1.0 + np.exp(-logits))
    p = np.clip(p, 1e-12, 1 - 1e-12)
    ll = np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
    return -2 * ll + df * np.log(n)


def fit_adaptive_lasso_class(X, y, lambda_grid, seed=0):
    beta_ridge = fit_ridge_stage1(X, y, seed=seed)
    weights = compute_adaptive_weights(beta_ridge)
    best = None
    for lam in lambda_grid:
        beta, intercept = fit_weighted_lasso_logistic(X, y, weights, lam)
        df = int(np.sum(np.abs(beta) > 1e-8)) + 1
        bic = bic_binary(X, y, beta, intercept, df)
        if best is None or bic < best['bic']:
            best = {'lam': lam, 'bic': bic, 'beta': beta, 'intercept': intercept}
    return best['beta'], best['intercept'], best['lam']


def run_adaptive_lasso_pipeline(Xd_train, Ylab_train, K, lambda_grid=None,
                                 tau=TAU_FALLBACK, seed=0):
    n, p = Xd_train.shape
    if lambda_grid is None:
        a_grid = np.linspace(0.1, 1.5, 20)
        lambda_grid = n ** (-a_grid)

    betas, active_sets = {}, {}
    for c in range(K):
        y_c = (Ylab_train == c).astype(int)
        beta_c, intercept_c, lam_c = fit_adaptive_lasso_class(Xd_train, y_c, lambda_grid, seed=seed)
        betas[c] = {'beta': beta_c, 'intercept': intercept_c, 'lambda': lam_c}
        active_sets[c] = set(np.where(np.abs(beta_c) > 1e-8)[0])

    A_intersection = set.intersection(*active_sets.values()) if active_sets else set()
    A_union = set.union(*active_sets.values()) if active_sets else set()
    freq = np.zeros(p, dtype=int)
    for c in range(K):
        for j in active_sets[c]:
            freq[j] += 1
    thresh = int(np.floor(K * tau))
    A_freq = set(np.where(freq >= thresh)[0])

    if len(A_intersection) > 0:
        A_hat, rule_used = A_intersection, 'intersection'
    elif len(A_freq) > 0:
        A_hat, rule_used = A_freq, 'frequency_fallback'
    else:
        A_hat, rule_used = A_union, 'union'

    return {'betas': betas, 'active_sets': active_sets, 'A_hat': A_hat, 'rule_used': rule_used}


def fit_final_multinomial(Xd_train, Ylab_train, A_hat):
    if len(A_hat) == 0:
        A_hat = set(range(Xd_train.shape[1]))
    cols = sorted(A_hat)
    X_sub = Xd_train[:, cols]
    model = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=5000)
    model.fit(X_sub, Ylab_train)
    return model, cols


def predict_adaptive_lasso(model, cols, Xd_test):
    return model.predict_proba(Xd_test[:, cols])


def derive_group_structure_from_columns(A_hat, groups):
    zero_g = set()
    for name, sl in groups:
        cols = set(range(sl.start, sl.stop))
        if cols.isdisjoint(A_hat):
            zero_g.add(name)
    return zero_g, set()


def coef_to_reference_parameterization(model, cols, K, p_total):
    coef_full = model.coef_
    B_rel_sub = coef_full[:K - 1, :] - coef_full[K - 1, :]
    B_full = np.zeros((K - 1, p_total))
    B_full[:, cols] = B_rel_sub
    return B_full


def fit_adaptive_lasso_full(Xd_train, Ylab_train, seed=0):
    t0 = time.time()
    pipeline_result = run_adaptive_lasso_pipeline(Xd_train, Ylab_train, K, seed=seed)
    model, cols = fit_final_multinomial(Xd_train, Ylab_train, pipeline_result['A_hat'])
    fit_time = time.time() - t0

    zero_g, tied_g = derive_group_structure_from_columns(pipeline_result['A_hat'], GROUPS)
    B_hat = coef_to_reference_parameterization(model, cols, K, Xd_train.shape[1])

    return {'model': model, 'cols': cols, 'A_hat': pipeline_result['A_hat'],
            'rule_used': pipeline_result['rule_used'], 'zero_g': zero_g, 'tied_g': tied_g,
            'B_hat': B_hat, 'fit_time': fit_time}


def run(seed, n_train, n_test):
    train, test = generate_train_test(seed, n_train, n_test)
    Xd_tr, Y_tr, B_true = train['Xd'], train['Ylab'], train['B_true']
    Xd_te, Y_te = test['Xd'], test['Ylab']

    result = fit_adaptive_lasso_full(Xd_tr, Y_tr, seed=seed)
    probs_te = predict_adaptive_lasso(result['model'], result['cols'], Xd_te)

    pred_metrics = evaluate_predictions(probs_te, Y_te)
    struct_metrics = evaluate_structure(result['zero_g'], result['tied_g'])
    coef_metrics = evaluate_coefficients(result['B_hat'], B_true)

    return {
        'method': 'Adaptive Lasso',
        'prediction_metrics': pred_metrics,
        'structure_metrics': struct_metrics,
        'coefficient_metrics': coef_metrics,
        'fit_time': result['fit_time'],
        'extra': {'A_hat': sorted(result['A_hat']), 'rule_used': result['rule_used']},
    }


if __name__ == "__main__":
    print("--- Running Adaptive Lasso Pipeline (v3, shared dgp.py) ---")
    res = run(seed=0, n_train=4000, n_test=4000)
    print("A_hat (selected columns):", res['extra']['A_hat'], " rule_used:", res['extra']['rule_used'])
    print("fit_time:", round(res['fit_time'], 2), "s")
    print("Prediction metrics:", res['prediction_metrics'])
    print("Structure metrics:", res['structure_metrics'])
    print("Coefficient metrics (relative Frobenius error):",
          res['coefficient_metrics']['relative_frobenius_error'])