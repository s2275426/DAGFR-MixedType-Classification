import time
import numpy as np
from scipy.special import logsumexp
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from itertools import combinations

# ============================================================
# Global Hyperparameters 
# ============================================================
GAMMA = 1.0
RHO_ADMM = 1.0
ETA_SAFETY = 0.9
ADMM_MAX_ITER = 3000
ADMM_TOL = 1e-5
MONITOR_WINDOW = 15
EPS_ACTIVE = 1e-3
EPS_FUSION = 1e-3
VAL_FRACTION = 0.15
BIC_TOLERANCE_MARGIN = 0.01
RANDOM_STATE = 42
ALPHA_RELAX = 1.6


# ============================================================
# Data Loading and Group Structure 
# ============================================================
class GroupStructure:
    def __init__(self, group_boundaries, group_names):
        self.boundaries = [tuple(b) for b in group_boundaries]
        self.names = list(group_names)
        self.n_groups = len(self.boundaries)
        self.p_col = self.boundaries[-1][1]

    def slice(self, m):
        s, e = self.boundaries[m]
        return slice(int(s), int(e))

    def width(self, m):
        s, e = self.boundaries[m]
        return int(e) - int(s)


def load_npz_dataset(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    X_train, y_train_raw = data["X_train"], data["y_train"]
    X_test, y_test_raw = data["X_test"], data["y_test"]
    group_boundaries = data["group_boundaries"]
    group_names = data["group_names"]

    all_codes = np.unique(np.concatenate([y_train_raw, y_test_raw]))
    K = len(all_codes)
    code_to_idx = {c: i for i, c in enumerate(all_codes)}
    y_train = np.array([code_to_idx[c] for c in y_train_raw])
    y_test = np.array([code_to_idx[c] for c in y_test_raw])

    group_struct = GroupStructure(group_boundaries, group_names)
    return X_train, y_train, X_test, y_test, K, group_struct, all_codes


def one_hot(y, K):
    oh = np.zeros((len(y), K))
    oh[np.arange(len(y)), y] = 1.0
    return oh


# ============================================================
# Multinomial Logistic Regression: Negative Log-Likelihood / Gradient 
# ============================================================
def _logits_full(B, b0, X):
    logits_free = X @ B.T + b0
    n = X.shape[0]
    return np.hstack([logits_free, np.zeros((n, 1))])


def nll(B, b0, X, y_onehot, K):
    n = X.shape[0]
    logits_full = _logits_full(B, b0, X)
    log_probs = logits_full - logsumexp(logits_full, axis=1, keepdims=True)
    return -np.sum(y_onehot * log_probs) / n


def grad(B, b0, X, y_onehot, K):
    n = X.shape[0]
    logits_full = _logits_full(B, b0, X)
    log_probs = logits_full - logsumexp(logits_full, axis=1, keepdims=True)
    probs = np.exp(log_probs)
    grad_full = (probs - y_onehot) / n
    grad_free = grad_full[:, :K - 1]
    grad_B = grad_free.T @ X
    grad_b0 = grad_free.sum(axis=0)
    return grad_B, grad_b0


# ============================================================
# Step 1: Joint Initial MLE 
# ============================================================
def fit_joint_mle(X, y, K, C_large=1e6):
    clf = LogisticRegression(penalty="l2", C=C_large, solver="lbfgs",
                              max_iter=3000, multi_class="multinomial")
    clf.fit(X, y)
    classes_sorted = clf.classes_
    ref_idx = len(classes_sorted) - 1

    coef_full = clf.coef_
    intercept_full = clf.intercept_

    B_tilde = coef_full[:ref_idx, :] - coef_full[ref_idx:ref_idx + 1, :]
    b0_tilde = intercept_full[:ref_idx] - intercept_full[ref_idx]
    return B_tilde, b0_tilde


def compute_weights(B_tilde, group_struct, K, eps_n, gamma=GAMMA):
    n_groups = group_struct.n_groups
    w_sparse = np.zeros(n_groups)
    for m in range(n_groups):
        sl = group_struct.slice(m)
        frob = np.linalg.norm(B_tilde[:, sl], ord="fro")
        w_sparse[m] = (frob + eps_n) ** (-gamma)

    pairs = list(combinations(range(K - 1), 2))
    w_fusion = {}
    for m in range(n_groups):
        sl = group_struct.slice(m)
        for (c, c2) in pairs:
            d = np.linalg.norm(B_tilde[c, sl] - B_tilde[c2, sl], ord=2)
            w_fusion[(m, c, c2)] = (d + eps_n) ** (-gamma)
    return w_sparse, w_fusion, pairs


# ============================================================
# Step 3: ADMM Solver
# ============================================================
def group_soft_threshold(v, thresh):
    nv = np.linalg.norm(v, ord=2)
    if nv <= thresh:
        return np.zeros_like(v)
    return (1 - thresh / nv) * v


def dagfr_admm(X, y_onehot, K, group_struct, lambda_P, lambda_F,
               w_sparse, w_fusion, pairs,
               B_init=None, b0_init=None, z_init=None, u_init=None,
               rho=RHO_ADMM, eta_safety=ETA_SAFETY,
               max_iter=ADMM_MAX_ITER, tol=ADMM_TOL,
               alpha_relax=ALPHA_RELAX,
               verbose_convergence=False):
    n, p_col = X.shape
    n_groups = group_struct.n_groups

    B = np.zeros((K - 1, p_col)) if B_init is None else B_init.copy()
    b0 = np.zeros(K - 1) if b0_init is None else b0_init.copy()

    z = {key: np.zeros(group_struct.width(key[0])) for key in w_fusion} \
        if z_init is None else {k: v.copy() for k, v in z_init.items()}
    u = {key: np.zeros_like(z[key]) for key in z} \
        if u_init is None else {k: v.copy() for k, v in u_init.items()}

    XtX = X.T @ X / n
    L_max_eig = np.linalg.eigvalsh(XtX).max()
    L_smooth = 0.25 * L_max_eig + rho * max(K - 2, 1)
    eta = eta_safety / L_smooth

    converged = False
    primal_res_norm, step_size = np.inf, np.inf

    for t in range(max_iter):
        grad_B, grad_b0 = grad(B, b0, X, y_onehot, K)
        grad_admm = np.zeros_like(B)
        for (m, c, c2), _ in w_fusion.items():
            sl = group_struct.slice(m)
            r = B[c, sl] - B[c2, sl] - z[(m, c, c2)] + u[(m, c, c2)]
            grad_admm[c, sl] += rho * r
            grad_admm[c2, sl] -= rho * r

        B_tilde_step = B - eta * (grad_B + grad_admm)
        b0_new = b0 - eta * grad_b0

        B_new = B_tilde_step.copy()
        for c in range(K - 1):
            for m in range(n_groups):
                sl = group_struct.slice(m)
                B_new[c, sl] = group_soft_threshold(B_tilde_step[c, sl],
                                                     eta * lambda_P * w_sparse[m])

        z_new, u_new = {}, {}
        primal_res_sq = 0.0
        for (m, c, c2), w_val in w_fusion.items():
            sl = group_struct.slice(m)
            B_hat_c = alpha_relax * B_new[c, sl] + (1 - alpha_relax) * B[c, sl]
            B_hat_c2 = alpha_relax * B_new[c2, sl] + (1 - alpha_relax) * B[c2, sl]
            v = B_hat_c - B_hat_c2 + u[(m, c, c2)]
            thresh = lambda_F * w_val / rho
            z_val_new = group_soft_threshold(v, thresh)
            z_new[(m, c, c2)] = z_val_new
            r = (B_hat_c - B_hat_c2) - z_val_new
            u_new[(m, c, c2)] = u[(m, c, c2)] + r
            primal_res_sq += np.sum(r ** 2)

        primal_res_norm = np.sqrt(primal_res_sq)
        step_size = np.linalg.norm(B_new - B)

        B, b0, z, u = B_new, b0_new, z_new, u_new

        if primal_res_norm < tol and step_size < tol:
            converged = True
            break

    if verbose_convergence:
        status = "✓ Converged" if converged else "✗ Not Converged (Max Iter Reached)"
        print(f"      [ADMM Diagnostic] {status}  Final Iter={t+1}  "
              f"primal_res={primal_res_norm:.2e}  step_size={step_size:.2e}")

    return B, b0, z, u, t + 1, converged


# ============================================================
# Step 4: Structure Detection
# ============================================================
def detect_structure(B, group_struct, K, eps_active=EPS_ACTIVE, eps_fusion=EPS_FUSION):
    n_groups = group_struct.n_groups
    group_active = np.zeros(n_groups, dtype=bool)
    fusion_clusters = {}

    for m in range(n_groups):
        sl = group_struct.slice(m)
        max_norm = max(np.linalg.norm(B[c, sl], ord=2) for c in range(K - 1))
        group_active[m] = max_norm > eps_active

        parent = list(range(K))
        def find(x):
            while parent[x] != x:
                x = parent[x]
            return x
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        if not group_active[m]:
            for c in range(K - 1):
                union(c, K - 1)
        else:
            for c in range(K - 1):
                if np.linalg.norm(B[c, sl], ord=2) <= eps_fusion:
                    union(c, K - 1)
            for c, c2 in combinations(range(K - 1), 2):
                if np.linalg.norm(B[c, sl] - B[c2, sl], ord=2) <= eps_fusion:
                    union(c, c2)

        clusters_map = {}
        for c in range(K):
            clusters_map.setdefault(find(c), []).append(c)
        fusion_clusters[m] = list(clusters_map.values())

    return group_active, fusion_clusters


# ============================================================
# Step 4: Constrained Refit 
# ============================================================
def _build_param_map(group_struct, K, fusion_clusters):
    param_blocks = []
    for m in range(group_struct.n_groups):
        for cluster in fusion_clusters[m]:
            if (K - 1) in cluster:
                continue
            param_blocks.append((m, cluster, group_struct.width(m)))
    total_dim = sum(w for _, _, w in param_blocks)
    return param_blocks, total_dim


def _expand(theta, param_blocks, group_struct, K):
    B = np.zeros((K - 1, group_struct.p_col))
    ptr = 0
    for (m, cluster, w) in param_blocks:
        sl = group_struct.slice(m)
        block = theta[ptr:ptr + w]
        for c in cluster:
            if c != K - 1:
                B[c, sl] = block
        ptr += w
    return B


def _collapse_grad(grad_B, param_blocks, group_struct, K):
    theta_grad = np.zeros(sum(w for _, _, w in param_blocks))
    ptr = 0
    for (m, cluster, w) in param_blocks:
        sl = group_struct.slice(m)
        acc = np.zeros(w)
        for c in cluster:
            if c != K - 1:
                acc += grad_B[c, sl]
        theta_grad[ptr:ptr + w] = acc
        ptr += w
    return theta_grad


def refit_with_structure(X, y_onehot, K, group_struct, group_active, fusion_clusters):
    param_blocks, total_dim = _build_param_map(group_struct, K, fusion_clusters)
    n_intercepts = K - 1
    df = total_dim

    def objective(full_theta):
        theta_B = full_theta[:total_dim]
        b0 = full_theta[total_dim:]
        B = _expand(theta_B, param_blocks, group_struct, K)
        f = nll(B, b0, X, y_onehot, K)
        gB, gb0 = grad(B, b0, X, y_onehot, K)
        g_theta = _collapse_grad(gB, param_blocks, group_struct, K)
        return f, np.concatenate([g_theta, gb0])

    x0 = np.zeros(total_dim + n_intercepts)
    res = minimize(objective, x0, jac=True, method="L-BFGS-B",
                    options={"maxiter": 500, "ftol": 1e-10})

    theta_B_opt = res.x[:total_dim]
    b0_opt = res.x[total_dim:]
    B_refit = _expand(theta_B_opt, param_blocks, group_struct, K)
    return B_refit, b0_opt, df + n_intercepts


def compute_bic(X, y_onehot, K, B_refit, b0_refit, df, n):
    ll = -nll(B_refit, b0_refit, X, y_onehot, K) * n
    return -2 * ll + df * np.log(n)


def eval_acc(B_refit, b0_refit, X, y):
    logits = _logits_full(B_refit, b0_refit, X)
    pred = np.argmax(logits, axis=1)
    return np.mean(pred == y)


def eval_pred(B_refit, b0_refit, X):
    logits = _logits_full(B_refit, b0_refit, X)
    return np.argmax(logits, axis=1)


# ============================================================
# BIC Tolerance Margin Selection 
# ============================================================
def select_lambda_within_bic_margin(results, margin=BIC_TOLERANCE_MARGIN):
    bic_min = min(r["bic"] for r in results)
    threshold = bic_min * (1 + margin) if bic_min > 0 else bic_min * (1 - margin)
    candidates = [r for r in results if r["bic"] <= threshold]
    best = max(candidates, key=lambda r: r["val_acc"])
    print(f"    [Tolerance Selection] BIC_min={bic_min:.2f}  Threshold={threshold:.2f}  "
          f"Candidates={len(candidates)}/{len(results)}")
    for r in candidates:
        marker = " ← Selected" if r is best else ""
        print(f"      λ={r['lam']:.5f}  BIC={r['bic']:.2f}  df={r['df']:<4d}  "
              f"val_acc={r['val_acc']:.4f}{marker}")
    return best


# ============================================================
# Local Joint Refinement (Cold Start, Unchanged)
# ============================================================
def local_joint_refinement(X_fit, y_fit_oh, X_val, y_val, K, group_struct,
                            w_sparse, w_fusion, pairs, n_fit,
                            lambda_P_center, lambda_F_center,
                            n_local_points=5, span_factor=4.0,
                            bic_margin_wide=0.06):
    lp_grid = np.logspace(np.log10(lambda_P_center / span_factor),
                           np.log10(lambda_P_center * span_factor), n_local_points)
    lf_grid = np.logspace(np.log10(max(lambda_F_center / span_factor, 1e-5)),
                           np.log10(lambda_F_center * span_factor), n_local_points)

    results = []
    for lam_P in lp_grid:
        for lam_F in lf_grid:
            B, b0, z, u, n_iter, converged = dagfr_admm(
                X_fit, y_fit_oh, K, group_struct, lam_P, lam_F,
                w_sparse, w_fusion, pairs,
                B_init=None, b0_init=None, z_init=None, u_init=None,
                max_iter=ADMM_MAX_ITER, tol=ADMM_TOL, verbose_convergence=False
            )
            active, clusters = detect_structure(B, group_struct, K)
            B_refit, b0_refit, df = refit_with_structure(X_fit, y_fit_oh, K,
                                                            group_struct, active, clusters)
            bic = compute_bic(X_fit, y_fit_oh, K, B_refit, b0_refit, df, n_fit)
            val_acc = eval_acc(B_refit, b0_refit, X_val, y_val)
            results.append({"lam_P": lam_P, "lam_F": lam_F, "bic": bic,
                             "df": df, "val_acc": val_acc})
            print(f"    λ_P={lam_P:.5f}  λ_F={lam_F:.5f}  BIC={bic:.2f}  "
                  f"df={df:<4d}  val_acc={val_acc:.4f}")

    bic_min = min(r["bic"] for r in results)
    threshold = bic_min * (1 + bic_margin_wide)
    candidates = [r for r in results if r["bic"] <= threshold]
    best = max(candidates, key=lambda r: r["val_acc"])
    print(f"    [Joint Refinement-Margin={bic_margin_wide*100:.0f}%] "
          f"Candidates={len(candidates)}/{len(results)}  "
          f"Selected: λ_P={best['lam_P']:.5f}  λ_F={best['lam_F']:.5f}  "
          f"BIC={best['bic']:.2f}  val_acc={best['val_acc']:.4f}")
    return best


def build_lambda_grid():
    grid = np.concatenate([
        np.logspace(-5, np.log10(0.02), 6),
        np.logspace(np.log10(0.02), np.log10(2.0), 12),
        np.logspace(np.log10(2.0), 1, 4),
    ])
    return np.unique(np.round(grid, 7))


# ============================================================
# DAGFR-Specific Metrics: Effective Feature Columns / Fusion Compression
# ============================================================
def compute_dagfr_unique_metrics(group_struct, K, active_final, clusters_final, df_final):
    n_groups = group_struct.n_groups
    n_zero_groups = int((~active_final).sum())
    n_active_groups = n_groups - n_zero_groups

    # Effective feature columns: Only counts raw columns occupied by "active" groups,
    # representing sensor/covariate channels needed during deployment (zero-groups require no measurement).
    effective_p_col = sum(group_struct.width(m) for m in range(n_groups) if active_final[m])
    total_p_col = group_struct.p_col

    # Additional parameter compression from fusion: For each active group,
    # (K-1) minus actual independent clusters is the number of free parameters saved purely by fusion.
    fusion_saved_params = 0
    total_fusion_merges = 0
    for m in range(n_groups):
        if not active_final[m]:
            continue
        # Exclude cluster containing reference class K-1 (ref class has no free parameters)
        n_free_blocks = sum(1 for cl in clusters_final[m] if (K - 1) not in cl)
        n_full_blocks = K - 1  # Unfused group has K-1 independent coefficient vectors
        fusion_saved_params += (n_full_blocks - n_free_blocks) * group_struct.width(m)
        total_fusion_merges += (n_full_blocks - n_free_blocks)

    return {
        "n_zero_groups": n_zero_groups,
        "n_active_groups": n_active_groups,
        "effective_p_col": effective_p_col,
        "total_p_col": total_p_col,
        "feature_reduction_ratio": 1 - effective_p_col / total_p_col,
        "fusion_saved_params": fusion_saved_params,
        "total_fusion_merges": total_fusion_merges,
    }


# ============================================================
# Main Execution Pipeline (M_k Path + Timing + Comprehensive Output)
# ============================================================
def run_dagfr(npz_path, label):
    print(f"\n{'='*74}")
    print(f"  DAGFR Execution: {label}  (Data Source: {npz_path})")
    print(f"{'='*74}")

    pipeline_t0 = time.time()

    X_train_full, y_train_full, X_test, y_test, K, group_struct, codes = \
        load_npz_dataset(npz_path)
    n_train_full, p_col = X_train_full.shape
    print(f"  n_train_full={n_train_full}, n_test={X_test.shape[0]}, "
          f"p_col={p_col}, |G|={group_struct.n_groups}, K={K}")

    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train_full, y_train_full, test_size=VAL_FRACTION,
        stratify=y_train_full, random_state=RANDOM_STATE
    )
    print(f"  Internal Split: n_fit={X_fit.shape[0]}, n_val={X_val.shape[0]} "
          f"(Test set n_test={X_test.shape[0]} is completely excluded from hyperparameter selection)")

    y_fit_oh = one_hot(y_fit, K)
    n_fit = X_fit.shape[0]
    eps_n = 1.0 / np.sqrt(n_fit)
    print(f"  eps_n = 1/sqrt(n_fit) = {eps_n:.6f}")

    data = np.load(npz_path, allow_pickle=True)
    if "Mk_per_channel" in data:
        print("  [Diagnostic] Per-channel M_k configuration:")
        for name, mk in zip(group_struct.names, data["Mk_per_channel"]):
            print(f"    {name:<22s}: M_k = {mk}")

    # ---- Step 1 ----
    print("\n  [Step 1] Fitting joint initial MLE (on fit subset)...")
    B_tilde, b0_tilde = fit_joint_mle(X_fit, y_fit, K)
    mle_acc = eval_acc(B_tilde, b0_tilde, X_fit, y_fit)
    print(f"  [Diagnostic] Unpenalized MLE fit set accuracy (num_params={(K-1)*p_col}) = {mle_acc:.4f}")

    w_sparse, w_fusion, pairs = compute_weights(B_tilde, group_struct, K, eps_n)
    print("  Group Frobenius norms and adaptive weights:")
    for m in range(group_struct.n_groups):
        sl = group_struct.slice(m)
        frob = np.linalg.norm(B_tilde[:, sl], ord="fro")
        print(f"    {group_struct.names[m]:<22s}: ‖B̃‖_F={frob:.4f}, ŵ_m={w_sparse[m]:.4f}")

    lambda_grid = build_lambda_grid()
    print(f"\n  λ Grid ({len(lambda_grid)} points): {np.round(lambda_grid, 5)}")

    # ---- Step 2a: λ_P Search (Cold Start) ----
    print(f"\n  [Step 2a] BIC Coordinate Search λ_P (Fixed λ_F=0, Cold Start)")
    t_2a_start = time.time()
    results_P = []
    for lam_P in lambda_grid:
        B, b0, z, u, n_iter, converged = dagfr_admm(
            X_fit, y_fit_oh, K, group_struct, lam_P, 0.0,
            w_sparse, w_fusion, pairs,
            B_init=None, b0_init=None, z_init=None, u_init=None,
            max_iter=ADMM_MAX_ITER, tol=ADMM_TOL, verbose_convergence=True
        )
        active, clusters = detect_structure(B, group_struct, K)
        B_refit, b0_refit, df = refit_with_structure(X_fit, y_fit_oh, K,
                                                        group_struct, active, clusters)
        bic = compute_bic(X_fit, y_fit_oh, K, B_refit, b0_refit, df, n_fit)
        val_acc = eval_acc(B_refit, b0_refit, X_val, y_val)
        n_active = int(active.sum())
        print(f"    λ_P={lam_P:.5f}  BIC={bic:.2f}  df={df:<4d}  "
              f"Active={n_active}/{group_struct.n_groups}  val_acc={val_acc:.4f}  "
              f"Iter={n_iter}  Converged={converged}")
        results_P.append({"lam": lam_P, "bic": bic, "df": df, "val_acc": val_acc,
                           "active": active, "clusters": clusters})
    t_2a_end = time.time()

    best_P = select_lambda_within_bic_margin(results_P)
    lambda_P_star = best_P["lam"]
    idx_P = list(lambda_grid).index(lambda_P_star) if lambda_P_star in lambda_grid else -1
    boundary_P = (idx_P == 0 or idx_P == len(lambda_grid) - 1)
    print(f"  → λ_P* = {lambda_P_star:.5f}  "
          f"{'[⚠️ Boundary Solution]' if boundary_P else '[Interior Solution]'}  "
          f"(Time: {t_2a_end - t_2a_start:.1f}s)")

    # ---- Step 2b: λ_F Search (Cold Start) ----
    print(f"\n  [Step 2b] BIC Coordinate Search λ_F (Fixed λ_P={lambda_P_star:.5f}, Cold Start)")
    t_2b_start = time.time()
    results_F = []
    for lam_F in lambda_grid:
        B, b0, z, u, n_iter, converged = dagfr_admm(
            X_fit, y_fit_oh, K, group_struct, lambda_P_star, lam_F,
            w_sparse, w_fusion, pairs,
            B_init=None, b0_init=None, z_init=None, u_init=None,
            max_iter=ADMM_MAX_ITER, tol=ADMM_TOL, verbose_convergence=True
        )
        active, clusters = detect_structure(B, group_struct, K)
        B_refit, b0_refit, df = refit_with_structure(X_fit, y_fit_oh, K,
                                                        group_struct, active, clusters)
        bic = compute_bic(X_fit, y_fit_oh, K, B_refit, b0_refit, df, n_fit)
        val_acc = eval_acc(B_refit, b0_refit, X_val, y_val)
        n_fused = sum(1 for m in range(group_struct.n_groups)
                      for cl in clusters[m] if len(cl) > 1)
        print(f"    λ_F={lam_F:.5f}  BIC={bic:.2f}  df={df:<4d}  "
              f"Groups with Fused Clusters={n_fused}  val_acc={val_acc:.4f}  "
              f"Iter={n_iter}  Converged={converged}")
        results_F.append({"lam": lam_F, "bic": bic, "df": df, "val_acc": val_acc,
                           "active": active, "clusters": clusters})
    t_2b_end = time.time()

    best_F = select_lambda_within_bic_margin(results_F)
    lambda_F_star = best_F["lam"]
    idx_F = list(lambda_grid).index(lambda_F_star) if lambda_F_star in lambda_grid else -1
    boundary_F = (idx_F == 0 or idx_F == len(lambda_grid) - 1)
    print(f"  → λ_F* = {lambda_F_star:.5f}  "
          f"{'[⚠️ Boundary Solution]' if boundary_F else '[Interior Solution]'}  "
          f"(Time: {t_2b_end - t_2b_start:.1f}s)")

    # ---- Step 2c: Local Joint Refinement (Cold Start) ----
    print(f"\n  [Step 2c] Local Joint Refinement centered at ({lambda_P_star:.5f}, {lambda_F_star:.5f})")
    t_2c_start = time.time()
    best_joint = local_joint_refinement(
        X_fit, y_fit_oh, X_val, y_val, K, group_struct,
        w_sparse, w_fusion, pairs, n_fit,
        lambda_P_center=lambda_P_star, lambda_F_center=lambda_F_star
    )
    t_2c_end = time.time()
    if best_joint["val_acc"] > best_F["val_acc"] + 0.005:
        print(f"    → Joint refinement found superior point, adopting: λ_P={best_joint['lam_P']:.5f}, "
              f"λ_F={best_joint['lam_F']:.5f} (val_acc {best_F['val_acc']:.4f} → "
              f"{best_joint['val_acc']:.4f})")
        lambda_P_star, lambda_F_star = best_joint["lam_P"], best_joint["lam_F"]
        boundary_P = False
        boundary_F = False
    else:
        print(f"    → Joint refinement found no significant gain, retaining coordinate search results")
    print(f"  (Step 2c Time: {t_2c_end - t_2c_start:.1f}s)")

    hyperparam_search_time = t_2a_end - t_2a_start + t_2b_end - t_2b_start + t_2c_end - t_2c_start

    # ---- Final Model: Re-fit with (λ_P*, λ_F*) on Full Training Set (Cold Start) ----
    print(f"\n  [Final Fit] Solving on full training set using (λ_P*, λ_F*) = ({lambda_P_star:.5f}, {lambda_F_star:.5f})...")
    t_final_start = time.time()
    y_train_full_oh = one_hot(y_train_full, K)
    B_final, b0_final, _, _, n_iter_final, converged_final = dagfr_admm(
        X_train_full, y_train_full_oh, K, group_struct, lambda_P_star, lambda_F_star,
        w_sparse, w_fusion, pairs, max_iter=ADMM_MAX_ITER, tol=ADMM_TOL,
        verbose_convergence=True
    )
    active_final, clusters_final = detect_structure(B_final, group_struct, K)
    B_refit_final, b0_refit_final, df_final = refit_with_structure(
        X_train_full, y_train_full_oh, K, group_struct, active_final, clusters_final
    )
    t_final_end = time.time()
    final_fit_time = t_final_end - t_final_start
    total_train_time = time.time() - pipeline_t0

    print(f"\n  [Structure Detection Results]")
    n_zero_groups = int((~active_final).sum())
    print(f"    Zero Groups = {n_zero_groups}/{group_struct.n_groups}")
    for m in range(group_struct.n_groups):
        status = "ZERO" if not active_final[m] else f"clusters={clusters_final[m]}"
        print(f"    {group_struct.names[m]:<22s}: {status}")

    # ---- Inference Time + Accuracy ----
    t_inf_start = time.time()
    test_pred = eval_pred(B_refit_final, b0_refit_final, X_test)
    t_inf_end = time.time()
    inference_time = t_inf_end - t_inf_start
    inference_time_per_sample = inference_time / X_test.shape[0]

    test_acc = np.mean(test_pred == y_test)
    train_pred = eval_pred(B_refit_final, b0_refit_final, X_train_full)
    train_acc = np.mean(train_pred == y_train_full)

    # ---- DAGFR-Specific Metrics ----
    unique_metrics = compute_dagfr_unique_metrics(
        group_struct, K, active_final, clusters_final, df_final
    )
    total_params_full = (K - 1) * p_col + (K - 1)
    compression_ratio = df_final / total_params_full

    # ---- Confusion Matrix / Classification Report ----
    cm = confusion_matrix(y_test, test_pred)
    report_str = classification_report(
        y_test, test_pred, target_names=[f"act={c}" for c in codes], digits=4
    )

    # ============================================================
    # Chapter 5 Complete Metrics Output
    # ============================================================
    print(f"\n{'-'*74}")
    print(f"  [Chapter 5 Comparison Metrics Summary - {label}]")
    print(f"{'-'*74}")
    print(f"  ── General Level (Directly comparable with Spline baseline) ──")
    print(f"    Training Set Accuracy  = {train_acc:.4f}")
    print(f"    Test Set Accuracy      = {test_acc:.4f}")
    print(f"    Train-Test Acc Gap     = {train_acc - test_acc:.4f}")
    print(f"    Hyperparameter Search  = {hyperparam_search_time:.1f} s "
          f"(Step2a={t_2a_end-t_2a_start:.1f}s + Step2b={t_2b_end-t_2b_start:.1f}s "
          f"+ Step2c={t_2c_end-t_2c_start:.1f}s)")
    print(f"    Final Model Fit Time   = {final_fit_time:.1f} s")
    print(f"    Total Train Time       = {total_train_time:.1f} s (Search + Final Fit)")
    print(f"    Inference Time (Full)  = {inference_time*1000:.2f} ms "
          f"({inference_time_per_sample*1e6:.2f} μs/sample)")
    print(f"    Degrees of Freedom df  = {df_final} / Upper Bound {total_params_full} "
          f"(Compression Ratio = {compression_ratio:.3f})")
    print(f"\n  ── DAGFR-Specific Layer (Reflects methodology value of group sparsity + fusion) ──")
    print(f"    λ_P* (Group Sparsity)  = {lambda_P_star:.5f}  "
          f"{'[Boundary Solution]' if boundary_P else '[Interior Solution]'}")
    print(f"    λ_F* (Group Fusion)    = {lambda_F_star:.5f}  "
          f"{'[Boundary Solution]' if boundary_F else '[Interior Solution]'}")
    print(f"    Zero / Total Groups    = {unique_metrics['n_zero_groups']}/{group_struct.n_groups}")
    print(f"    Effective Feature Cols = {unique_metrics['effective_p_col']} / {unique_metrics['total_p_col']} "
          f"(Deployment saves {unique_metrics['feature_reduction_ratio']*100:.1f}% of raw feature columns, "
          f"corresponding to omitting entire sensor channels)")
    print(f"    Fusion Compressed df   = {unique_metrics['fusion_saved_params']} "
          f"(Free parameters saved purely by fusion that group sparsity alone cannot achieve)")
    print(f"    Total Cross-Class Merges = {unique_metrics['total_fusion_merges']} "
          f"(Total pairwise class coefficient bindings across active groups)")
    print(f"\n  ── Per-Functional-Block Fusion Cluster Details ──")
    for m in range(group_struct.n_groups):
        if active_final[m]:
            n_free_blocks = sum(1 for cl in clusters_final[m] if (K - 1) not in cl)
            print(f"    {group_struct.names[m]:<22s}: ACTIVE  "
                  f"Cluster Structure={clusters_final[m]}  (Independent Coef Blocks={n_free_blocks}/{K-1})")
        else:
            print(f"    {group_struct.names[m]:<22s}: ZERO (Entire block eliminated)")

    print(f"\n  [Test Set Confusion Matrix] (Rows = True Classes, Columns = Predicted Classes, Order = {codes.tolist()})")
    print(f"  {'':>6s}" + "".join(f"pred={c:<6}" for c in codes))
    for i, row in enumerate(cm):
        print(f"  true={codes[i]:<3}" + "".join(f"{v:<11d}" for v in row))

    print(f"\n  [Per-class Precision / Recall / F1-Score]")
    print("  " + report_str.replace("\n", "\n  "))
    print(f"{'-'*74}")

    return {
        "label": label, "p_col": p_col, "K": K,
        "lambda_P_star": lambda_P_star, "lambda_F_star": lambda_F_star,
        "boundary_P": boundary_P, "boundary_F": boundary_F,
        "df_final": df_final, "total_params_full": total_params_full,
        "compression_ratio": compression_ratio,
        "n_zero_groups": n_zero_groups, "n_groups": group_struct.n_groups,
        "effective_p_col": unique_metrics["effective_p_col"],
        "feature_reduction_ratio": unique_metrics["feature_reduction_ratio"],
        "fusion_saved_params": unique_metrics["fusion_saved_params"],
        "total_fusion_merges": unique_metrics["total_fusion_merges"],
        "train_acc": train_acc, "test_acc": test_acc,
        "hyperparam_search_time": hyperparam_search_time,
        "final_fit_time": final_fit_time,
        "total_train_time": total_train_time,
        "inference_time": inference_time,
        "inference_time_per_sample": inference_time_per_sample,
        "confusion_matrix": cm, "activity_codes": codes,
    }


if __name__ == "__main__":
    DESIGN_DIR = "/Users/augleovo/PycharmProjects/Application_New_副本/Experiment/design_matrices"

    # Per final decision, run only per-channel M_k path
    result = run_dagfr(f"{DESIGN_DIR}/design_matrix_per_channel_Mk.npz", "DAGFR (Per-channel M_k)")

    print(f"\n\n{'='*74}")
    print(f"  [For Chapter 5 Usage] DAGFR Final Metrics Overview")
    print(f"{'='*74}")
    for k, v in result.items():
        if k in ("confusion_matrix",):
            continue
        print(f"  {k:<28s}: {v}")
    print(f"{'='*74}")