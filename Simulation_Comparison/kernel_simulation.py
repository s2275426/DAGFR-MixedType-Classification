"""
kernel_simulation.py
Weighted Kernel baseline (Selk 2023).
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