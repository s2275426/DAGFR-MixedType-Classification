"""
DAGFR Comparison Harness v4: DAGFR Fit with Two-Stage BIC Selection
"""

import time
import numpy as np
from scipy.optimize import minimize

import dgp

K = dgp.K
P_TOTAL = dgp.P_TOTAL
GROUPS = dgp.GROUPS
softmax_probs = dgp.softmax_probs
generate_train_test = dgp.generate_train_test
evaluate_predictions = dgp.evaluate_predictions
evaluate_structure = dgp.evaluate_structure
evaluate_coefficients = dgp.evaluate_coefficients

GAMMA = 1.0
EPS_ACTIVE = 1e-6
EPS_FUSION_REL = 0.02  # 改为相对阈值，应对高相关性下的系数量级变化

ADMM_ETA_SAFETY = 0.4
OSC_CHECK_WINDOW = 200
OSC_CHECK_EVERY = 200
OSC_RATIO_THRESHOLD = 0.98
MAX_ETA_HALVINGS = 6
ADMM_MAX_ITER = 60000  # 从 20000 上调，给高相关设计矩阵更多收敛预算
ADMM_TOL = 1e-8


def eps_n(n, c=0.1):
    return c * n ** (-1.0)


def nll_and_grad_matrix(B, Xd, Ylab, K):
    Kc = K - 1
    n_ = Xd.shape[0]
    logits = np.hstack([Xd @ B.T, np.zeros((n_, 1))])
    logits -= logits.max(axis=1, keepdims=True)
    ex = np.exp(logits)
    probs = ex / ex.sum(axis=1, keepdims=True)
    onehot = np.zeros((n_, K))
    onehot[np.arange(n_), Ylab] = 1
    loss = -np.sum(onehot * np.log(probs + 1e-12)) / n_
    grad = (probs[:, :Kc] - onehot[:, :Kc]).T @ Xd / n_
    return loss, grad


def fit_mle(Xd, Ylab, K, p_total):
    def nll_flat(beta_flat):
        Kc = K - 1
        B = beta_flat.reshape(Kc, p_total)
        loss, grad = nll_and_grad_matrix(B, Xd, Ylab, K)
        return loss, grad.ravel()
    res = minimize(nll_flat, np.zeros((K - 1) * p_total), jac=True,
                   method='L-BFGS-B', options={'maxiter': 2000})
    return res.x.reshape(K - 1, p_total), res.success


def build_weights(B_mle, groups, mode, gamma=GAMMA, eps=1e-8):
    omega_hat, tau_hat = {}, {}
    for name, sl in groups:
        norm_joint = np.linalg.norm(B_mle[:, sl])
        norm_diff = np.linalg.norm(B_mle[0, sl] - B_mle[1, sl])
        if mode == 'uniform':
            omega_hat[name] = 1.0; tau_hat[name] = 1.0
        elif mode == 'adaptive_sparse':
            omega_hat[name] = (norm_joint + eps) ** (-gamma); tau_hat[name] = 1.0
        elif mode == 'doubly_adaptive':
            omega_hat[name] = (norm_joint + eps) ** (-gamma)
            tau_hat[name] = (norm_diff + eps) ** (-gamma)
        else:
            raise ValueError(mode)
    return omega_hat, tau_hat


def group_soft_threshold(v, thresh):
    norm_v = np.linalg.norm(v)
    if norm_v <= thresh:
        return np.zeros_like(v)
    return v * (1 - thresh / norm_v)


def run_fused_admm(Xd, Ylab, K, groups, lambda_P, lambda_F, omega_hat, tau_hat,
                   rho=4.0, max_iter=ADMM_MAX_ITER, tol=ADMM_TOL,
                   eta_safety=ADMM_ETA_SAFETY, B_init=None, Z_init=None, U_init=None):
    p = Xd.shape[1]
    Kc = K - 1
    B = np.zeros((Kc, p)) if B_init is None else B_init.copy()
    eigmax = np.linalg.eigvalsh(Xd.T @ Xd / Xd.shape[0]).max()
    L_smooth = 0.25 * eigmax + rho * max(K - 2, 0)
    eta = eta_safety / L_smooth

    Z = {name: np.zeros(sl.stop - sl.start) for name, sl in groups} if Z_init is None \
        else {k: v.copy() for k, v in Z_init.items()}
    U = {name: np.zeros(sl.stop - sl.start) for name, sl in groups} if U_init is None \
        else {k: v.copy() for k, v in U_init.items()}

    bchange_history = []
    n_eta_halvings = 0
    it = 0
    converged_flag = False

    while it < max_iter:
        loss, grad = nll_and_grad_matrix(B, Xd, Ylab, K)
        for name, sl in groups:
            diff = B[0, sl] - B[1, sl] - Z[name] + U[name]
            grad[0, sl] += rho * diff
            grad[1, sl] -= rho * diff

        B_temp = B - eta * grad
        B_new = np.zeros_like(B_temp)
        for i in range(Kc):
            for name, sl in groups:
                thresh = eta * lambda_P * omega_hat[name]
                B_new[i, sl] = group_soft_threshold(B_temp[i, sl], thresh)

        B_prev = B
        B = B_new
        B_change = np.linalg.norm(B - B_prev)
        bchange_history.append(B_change)

        primal_res_sq = 0.0
        for name, sl in groups:
            diff = B[0, sl] - B[1, sl]
            fusion_thresh = lambda_F * tau_hat[name] / rho
            Z_new = group_soft_threshold(diff + U[name], fusion_thresh)
            U[name] = U[name] + (diff - Z_new)
            primal_res_sq += np.linalg.norm(diff - Z_new) ** 2
            Z[name] = Z_new
        primal_res = np.sqrt(primal_res_sq)

        if B_change < tol and primal_res < tol:
            converged_flag = True
            break

        if (it + 1) % OSC_CHECK_EVERY == 0 and len(bchange_history) >= 2 * OSC_CHECK_WINDOW:
            recent = np.mean(bchange_history[-OSC_CHECK_WINDOW:])
            older = np.mean(bchange_history[-2 * OSC_CHECK_WINDOW:-OSC_CHECK_WINDOW])
            if older > 1e-12 and (recent / older) > OSC_RATIO_THRESHOLD and recent > tol:
                if n_eta_halvings < MAX_ETA_HALVINGS:
                    eta *= 0.5; n_eta_halvings += 1; bchange_history = []
        it += 1

    return B, Z, U, converged_flag, it


def detect_structure(B_admm, groups):
    """
    改为相对融合阈值：tied 判定基于 diff 相对于 joint norm 的比例，
    而不是绝对残差，这样在系数量级随相关性变化时判定更稳健。
    """
    zero_groups, tied_groups, free_groups = set(), set(), set()
    for name, sl in groups:
        norm0 = np.linalg.norm(B_admm[0, sl])
        norm1 = np.linalg.norm(B_admm[1, sl])
        if norm0 <= EPS_ACTIVE and norm1 <= EPS_ACTIVE:
            zero_groups.add(name); continue
        diff = np.linalg.norm(B_admm[0, sl] - B_admm[1, sl])
        joint = np.linalg.norm(B_admm[:, sl])
        rel_diff = diff / (joint + 1e-8)
        if rel_diff <= EPS_FUSION_REL:
            tied_groups.add(name)
        else:
            free_groups.add(name)
    return zero_groups, tied_groups, free_groups


def build_param_map(groups, tied_groups, zero_groups):
    entries = []
    idx = 0
    for name, sl in groups:
        dim = sl.stop - sl.start
        if name in zero_groups:
            entries.append(('zero', sl, None))
        elif name in tied_groups:
            entries.append(('tied', sl, slice(idx, idx + dim))); idx += dim
        else:
            entries.append(('row0', sl, slice(idx, idx + dim))); idx += dim
            entries.append(('row1', sl, slice(idx, idx + dim))); idx += dim
    return entries, idx


def theta_to_B(theta, entries, p, Kc):
    B = np.zeros((Kc, p))
    for kind, sl, tsl in entries:
        if kind == 'zero':
            continue
        val = theta[tsl]
        if kind == 'tied':
            B[0, sl] = val; B[1, sl] = val
        elif kind == 'row0':
            B[0, sl] = val
        elif kind == 'row1':
            B[1, sl] = val
    return B


def nll_reduced(theta, entries, p, Xd, Ylab, K):
    Kc = K - 1
    B = theta_to_B(theta, entries, p, Kc)
    loss, grad_full = nll_and_grad_matrix(B, Xd, Ylab, K)
    grad_theta = np.zeros_like(theta)
    for kind, sl, tsl in entries:
        if kind == 'zero':
            continue
        elif kind == 'tied':
            grad_theta[tsl] = grad_full[0, sl] + grad_full[1, sl]
        elif kind == 'row0':
            grad_theta[tsl] = grad_full[0, sl]
        elif kind == 'row1':
            grad_theta[tsl] = grad_full[1, sl]
    return loss, grad_theta


def refit(Xd, Ylab, K, groups, tied_groups, zero_groups):
    p = Xd.shape[1]
    entries, theta_size = build_param_map(groups, tied_groups, zero_groups)
    if theta_size == 0:
        return np.zeros((K - 1, p)), entries, theta_size, True
    res = minimize(nll_reduced, np.zeros(theta_size),
                   args=(entries, p, Xd, Ylab, K),
                   jac=True, method='L-BFGS-B', options={'maxiter': 2000})
    B_refit = theta_to_B(res.x, entries, p, K - 1)
    return B_refit, entries, theta_size, res.success


def bic_score(Xd, Ylab, B, df):
    n_ = Xd.shape[0]
    loss, _ = nll_and_grad_matrix(B, Xd, Ylab, K)
    return 2 * n_ * loss + np.log(n_) * df


def fit_dagfr(Xd_train, Ylab_train, lambda_grid=None):
    t0 = time.time()
    n = Xd_train.shape[0]
    if lambda_grid is None:
        a_grid = np.linspace(0.3, 1.2, 15)
        lambda_grid = n ** (-a_grid)

    B_mle, _ = fit_mle(Xd_train, Ylab_train, K, P_TOTAL)
    eps = eps_n(n)
    omega_hat, tau_hat = build_weights(B_mle, GROUPS, 'doubly_adaptive', eps=eps)

    def eval_lambda_pair(lam_P, lam_F, B_warm=None, Z_warm=None, U_warm=None):
        B_admm, Z_out, U_out, conv_flag, actual_iters = run_fused_admm(
            Xd_train, Ylab_train, K, GROUPS, lambda_P=lam_P, lambda_F=lam_F,
            omega_hat=omega_hat, tau_hat=tau_hat,
            B_init=B_warm, Z_init=Z_warm, U_init=U_warm)
        zero_g, tied_g, free_g = detect_structure(B_admm, GROUPS)
        B_refit, entries, theta_size, ok = refit(Xd_train, Ylab_train, K, GROUPS, tied_g, zero_g)
        bic = bic_score(Xd_train, Ylab_train, B_refit, df=theta_size)
        return bic, B_admm, Z_out, U_out, B_refit, zero_g, tied_g, free_g, theta_size, conv_flag, actual_iters

    best1 = None
    B_warm, Z_warm, U_warm = None, None, None
    for lam_P in lambda_grid:
        bic, B_admm, Z_warm, U_warm, *_, conv_flag, actual_iters = eval_lambda_pair(lam_P, 0.0, B_warm, Z_warm, U_warm)
        B_warm = B_admm
        if best1 is None or bic < best1['bic']:
            best1 = {'lam_P': lam_P, 'bic': bic}
    lambda_P_hat = best1['lam_P']

    best2 = None
    B_warm, Z_warm, U_warm = None, None, None
    for lam_F in lambda_grid:
        bic, B_admm, Z_warm, U_warm, B_refit, zero_g, tied_g, free_g, df, conv_flag, actual_iters = \
            eval_lambda_pair(lambda_P_hat, lam_F, B_warm, Z_warm, U_warm)
        B_warm = B_admm
        if best2 is None or bic < best2['bic']:
            best2 = {'lam_F': lam_F, 'bic': bic, 'B_refit': B_refit,
                     'zero_g': zero_g, 'tied_g': tied_g, 'free_g': free_g, 'df': df,
                     'admm_converged': conv_flag, 'admm_iters': actual_iters}
    lambda_F_hat = best2['lam_F']

    fit_time = time.time() - t0
    return {'B_hat': best2['B_refit'], 'zero_g': best2['zero_g'], 'tied_g': best2['tied_g'],
            'free_g': best2['free_g'], 'lambda_P_selected': lambda_P_hat,
            'lambda_F_selected': lambda_F_hat, 'fit_time': fit_time,
            'admm_converged': best2['admm_converged'], 'admm_iters': best2['admm_iters']}


def predict_dagfr(model, Xd_test):
    return softmax_probs(Xd_test, model['B_hat'])


def run(seed, n_train, n_test):
    train, test = generate_train_test(seed, n_train, n_test)
    Xd_tr, Y_tr, B_true = train['Xd'], train['Ylab'], train['B_true']
    Xd_te, Y_te = test['Xd'], test['Ylab']

    model = fit_dagfr(Xd_tr, Y_tr)
    probs_te = predict_dagfr(model, Xd_te)

    pred_metrics = evaluate_predictions(probs_te, Y_te)
    struct_metrics = evaluate_structure(model['zero_g'], model['tied_g'])
    coef_metrics = evaluate_coefficients(model['B_hat'], B_true)

    return {
        'method': 'DAGFR',
        'prediction_metrics': pred_metrics,
        'structure_metrics': struct_metrics,
        'coefficient_metrics': coef_metrics,
        'fit_time': model['fit_time'],
        'extra': {'lambda_P': model['lambda_P_selected'], 'lambda_F': model['lambda_F_selected'],
                  'zero_g': model['zero_g'], 'tied_g': model['tied_g'], 'free_g': model['free_g'],
                  'admm_converged': model['admm_converged'], 'admm_iters': model['admm_iters']},
    }


if __name__ == "__main__":
    print("--- Running DAGFR Pipeline (v4, shared dgp.py) ---")
    res = run(seed=0, n_train=4000, n_test=4000)
    print("lambda_P:", res['extra']['lambda_P'], " lambda_F:", res['extra']['lambda_F'],
          " fit_time:", round(res['fit_time'], 2), "s")
    print("ADMM Diagnostic -> Converged:", res['extra']['admm_converged'], " | Iterations:", res['extra']['admm_iters'])
    print("zero_g:", res['extra']['zero_g'], " tied_g:", res['extra']['tied_g'],
          " free_g:", res['extra']['free_g'])
    print("Prediction metrics:", res['prediction_metrics'])
    print("Structure metrics:", res['structure_metrics'])
    print("Coefficient metrics (relative Frobenius error):",
          res['coefficient_metrics']['relative_frobenius_error'])