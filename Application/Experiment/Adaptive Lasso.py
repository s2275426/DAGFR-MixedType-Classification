"""
adaptive_lasso.py
Strictly corresponds to Section 2.2, Algorithm 1: Two-Stage Adaptive Lasso Pipeline
for Mixed-Type Multi-Class Data. 
"""

import time
import numpy as np
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, classification_report
from joblib import Parallel, delayed
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# Global Hyperparameters (Algorithm 1)
# ============================================================
GAMMA_EXP = 1.0          # Paper fixes γ=1 (Eq 2.17)
V_FOLDS = 5              # V-fold CV for Ridge stage
TAU_FREQ = 0.5           # Frequency fallback threshold (Eq 2.23)
RIDGE_C_GRID = np.logspace(-3, 3, 13)   # Ridge stage C grid (equivalent to 1/λ_ridge)
FINAL_C = 1.0            # Final multinomial logistic regression fixed C=1.0 (Eq 2.26)
RANDOM_STATE = 42
N_JOBS = -1
EPS_DIV = None           # Filled with 1/sqrt(n) at runtime to prevent divide-by-zero in adaptive weights


def build_lambda_grid_adalasso():
    """Adaptive Lasso λ grid, covering the full range from light to strong regularization"""
    grid = np.concatenate([
        np.logspace(-4, np.log10(0.05), 6),
        np.logspace(np.log10(0.05), np.log10(2.0), 8),
        np.logspace(np.log10(2.0), 2, 4),
    ])
    return np.unique(np.round(grid, 6))


# ============================================================
# Data Loading and Group Structure (Shared with DAGFR/Spline npz, identical structure)
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


def classify_group_types(group_struct):
    """
    Identify the type of each functional block: continuous (weight/height/age) / dummy (gender) /
    functional (remaining 12 sensor channels, B-spline basis coefficient blocks).
    Corresponding to PDF Section 2.1: Continuous features and functional basis coefficients 
    require standardization, while dummy indicator variables remain on their original {0,1} scale (Eq 2.14).
    """
    types = []
    for name in group_struct.names:
        name_str = str(name)
        if name_str in ("weight", "height", "age"):
            types.append("continuous")
        elif name_str == "gender":
            types.append("dummy")
        else:
            types.append("functional")
    return types


def standardize_mixed(X_train, X_test, group_struct, group_types):
    """Corresponding to Eq 2.14: Standardize continuous + functional columns, preserve dummy columns as-is"""
    X_train_std = X_train.copy().astype(float)
    X_test_std = X_test.copy().astype(float)
    for m in range(group_struct.n_groups):
        if group_types[m] == "dummy":
            continue
        sl = group_struct.slice(m)
        mu = X_train[:, sl].mean(axis=0)
        sd = X_train[:, sl].std(axis=0)
        sd = np.where(sd < 1e-8, 1.0, sd)
        X_train_std[:, sl] = (X_train[:, sl] - mu) / sd
        X_test_std[:, sl] = (X_test[:, sl] - mu) / sd
    return X_train_std, X_test_std


# ============================================================
# Numerically Stable Binary Log-Likelihood (Used for BIC calculation, Eq 2.18)
# ============================================================
def binary_loglik(X, y, beta, beta0):
    z = X @ beta + beta0
    log1p_exp = np.maximum(z, 0) + np.log1p(np.exp(-np.abs(z)))
    return np.sum(y * z - log1p_exp)


def compute_bic_binary(X, y, beta, beta0, n):
    df = int(np.sum(np.abs(beta) > 1e-8)) + 1   # +1 for intercept
    ll = binary_loglik(X, y, beta, beta0)
    return -2 * ll + df * np.log(n), df


# ============================================================
# Single OvR Subproblem: Stage 1 (Ridge) + Stage 2 (Adaptive Lasso via BIC)
# ============================================================
def fit_adaptive_lasso_one_class(class_idx, class_label, X_fit, y_multiclass,
                                   lambda_grid, eps_n, v_folds=V_FOLDS):
    t0 = time.time()
    n_fit, p_col = X_fit.shape
    y_c = (y_multiclass == class_idx).astype(int)

    # ---- Stage 1: Ridge initial estimation (Eq 2.16), V-fold CV to select ridge strength ----
    t_ridge0 = time.time()
    kf = KFold(n_splits=v_folds, shuffle=True, random_state=RANDOM_STATE)
    ridge_cv = LogisticRegressionCV(
        Cs=RIDGE_C_GRID, cv=kf, penalty="l2", solver="lbfgs",
        scoring="neg_log_loss", max_iter=2000, n_jobs=1
    )
    ridge_cv.fit(X_fit, y_c)
    beta_tilde = ridge_cv.coef_.ravel()
    C_ridge_star = ridge_cv.C_[0]
    t_ridge1 = time.time()

    # ---- Stage 2: Adaptive weights construction (Eq 2.17, γ=1) ----
    w_j = 1.0 / (np.abs(beta_tilde) + eps_n) ** GAMMA_EXP
    X_rescaled = X_fit / w_j[None, :]     # Column rescaling trick

    # ---- Warm-start solver along regularization path, from large λ to small λ (Eq 2.19) ----
    t_lasso0 = time.time()
    clf = LogisticRegression(
        penalty="l1", solver="saga", warm_start=True,
        max_iter=4000, tol=1e-4
    )
    results = []
    for lam in sorted(lambda_grid, reverse=True):
        C_equiv = 1.0 / (n_fit * lam)
        clf.set_params(C=C_equiv)
        clf.fit(X_rescaled, y_c)
        gamma_hat = clf.coef_.ravel()
        beta0_hat = clf.intercept_[0]
        beta_hat = gamma_hat / w_j          # Recover true coefficients

        bic, df = compute_bic_binary(X_fit, y_c, beta_hat, beta0_hat, n_fit)
        converged = clf.n_iter_[0] < clf.max_iter
        results.append({
            "lam": lam, "beta": beta_hat.copy(), "beta0": beta0_hat,
            "bic": bic, "df": df, "converged": converged
        })
    t_lasso1 = time.time()

    best = min(results, key=lambda r: r["bic"])
    active_set = set(np.where(np.abs(best["beta"]) > 1e-6)[0].tolist())
    n_non_converged = sum(1 for r in results if not r["converged"])

    t1 = time.time()
    return {
        "class_idx": class_idx, "class_label": class_label,
        "C_ridge_star": C_ridge_star, "lambda_star": best["lam"],
        "df_star": best["df"], "bic_star": best["bic"],
        "beta": best["beta"], "beta0": best["beta0"],
        "active_set": active_set,
        "ridge_time": t_ridge1 - t_ridge0,
        "lasso_search_time": t_lasso1 - t_lasso0,
        "n_lambda_nonconverged": n_non_converged,
        "n_lambda_total": len(results),
        "total_time": t1 - t0,
    }


# ============================================================
# Multi-Class Aggregation: Intersection → Frequency Fallback → Union (Eq 2.21-2.23, Algorithm 1 line 11-19)
# ============================================================
def aggregate_feature_sets(class_results, p_col, K, tau=TAU_FREQ):
    active_sets = [r["active_set"] for r in class_results]

    A_intersection = set.intersection(*active_sets) if active_sets else set()
    A_union = set.union(*active_sets) if active_sets else set()

    freq_count = np.zeros(p_col, dtype=int)
    for s in active_sets:
        for j in s:
            freq_count[j] += 1
    threshold = int(np.floor(K * tau))
    A_freq = set(np.where(freq_count >= threshold)[0].tolist())

    if len(A_intersection) > 0:
        chosen, rule = A_intersection, "Intersection"
    elif len(A_freq) > 0:
        chosen, rule = A_freq, f"Frequency Fallback (≥{threshold}/{K} subproblems)"
    else:
        chosen, rule = A_union, "Union Fallback"

    return {
        "A_intersection": A_intersection, "A_freq": A_freq, "A_union": A_union,
        "chosen_set": chosen, "rule_used": rule, "freq_count": freq_count,
    }


# ============================================================
# Main Pipeline
# ============================================================
def run_adaptive_lasso(npz_path, label):
    print(f"\n{'='*74}")
    print(f"  Adaptive Lasso (Algorithm 1) Run: {label}")
    print(f"  Data source: {npz_path}")
    print(f"{'='*74}")

    pipeline_t0 = time.time()

    X_train_full, y_train_full, X_test, y_test, K, group_struct, codes = \
        load_npz_dataset(npz_path)
    n_train, p_col = X_train_full.shape
    print(f"  n_train={n_train}, n_test={X_test.shape[0]}, p_col={p_col}, "
          f"|G|={group_struct.n_groups}, K={K}")

    group_types = classify_group_types(group_struct)
    print("  Functional Block Type Identification:")
    for m in range(group_struct.n_groups):
        print(f"    {group_struct.names[m]:<22s}: Type={group_types[m]:<12s} "
              f"Width={group_struct.width(m)}")

    # ---- Step 1: Standardization (Algorithm 1, line 2) ----
    X_train_std, X_test_std = standardize_mixed(X_train_full, X_test,
                                                  group_struct, group_types)
    eps_n = 1.0 / np.sqrt(n_train)
    print(f"\n  eps_n = 1/sqrt(n_train) = {eps_n:.6f} (adaptive weight divide-by-zero protection)")

    lambda_grid = build_lambda_grid_adalasso()
    print(f"  λ Grid ({len(lambda_grid)} points): {np.round(lambda_grid, 5)}")

    # ---- Step 2: Parallel solution of K OvR subproblems (Algorithm 1, line 3-10) ----
    print(f"\n  [Stage 1+2] Parallel solving Ridge initial estimation + Adaptive Lasso BIC search "
          f"across {K} OvR subproblems...")
    t_ovr0 = time.time()
    class_results = Parallel(n_jobs=N_JOBS, backend="loky")(
        delayed(fit_adaptive_lasso_one_class)(
            c, codes[c], X_train_std, y_train_full, lambda_grid, eps_n
        ) for c in range(K)
    )
    t_ovr1 = time.time()

    print(f"\n  {'Class':<10s}{'C_ridge*':<12s}{'λ_c*':<12s}{'df':<8s}"
          f"{'|A_c|':<10s}{'BIC':<14s}{'Non-converged λ points':<14s}")
    for r in class_results:
        print(f"  act={r['class_label']:<6}{r['C_ridge_star']:<12.4f}"
              f"{r['lambda_star']:<12.5f}{r['df_star']:<8d}"
              f"{len(r['active_set']):<10d}{r['bic_star']:<14.2f}"
              f"{r['n_lambda_nonconverged']}/{r['n_lambda_total']}")

    # ---- Step 3: Feature Aggregation (Algorithm 1, line 11-19) ----
    agg = aggregate_feature_sets(class_results, p_col, K)
    A_hat = sorted(agg["chosen_set"])
    print(f"\n  [Feature Aggregation]")
    print(f"    |Intersection A∩| = {len(agg['A_intersection'])}")
    print(f"    |Frequency Fallback Set (τ={TAU_FREQ})| = {len(agg['A_freq'])}")
    print(f"    |Union A∪| = {len(agg['A_union'])}")
    print(f"    → Applied Rule: {agg['rule_used']}  Final |Â| = {len(A_hat)}")

    # ---- Feature selection statistics per functional block (Comparison with DAGFR group-level all-or-nothing sparsity) ----
    print(f"\n  [Selection per Functional Block] (Adaptive Lasso performs column-level selection rather than group-level selection)")
    group_selection_detail = []
    for m in range(group_struct.n_groups):
        sl = group_struct.slice(m)
        cols = set(range(sl.start, sl.stop))
        n_selected = len(cols & set(A_hat))
        width = group_struct.width(m)
        group_selection_detail.append({
            "name": group_struct.names[m], "width": width,
            "n_selected": n_selected,
        })
        status = "ZERO (entire block unselected)" if n_selected == 0 else \
                 ("ALL (fully selected)" if n_selected == width else "PARTIAL (partially selected)")
        print(f"    {group_struct.names[m]:<22s}: {n_selected:>3d}/{width:<3d} "
              f"columns selected  [{status}]")

    n_fully_zero_groups = sum(1 for g in group_selection_detail if g["n_selected"] == 0)
    n_partial_groups = sum(1 for g in group_selection_detail
                            if 0 < g["n_selected"] < g["width"])
    n_full_groups = sum(1 for g in group_selection_detail
                         if g["n_selected"] == g["width"])

    # ---- Feature Importance (Eq 2.22, for visualization only, not used for selection) ----
    importance = np.zeros(p_col)
    for r in class_results:
        importance += np.abs(r["beta"])
    importance /= K
    top_idx = np.argsort(-importance)[:15]
    print(f"\n  [Feature Importance Top-15] (Eq 2.22, for visualization purposes only, does not affect feature selection)")
    for j in top_idx:
        print(f"    Col {j:<5d}  Importance={importance[j]:.4f}")

    # ---- Step 4: Final Multinomial Logistic Classifier (Eq 2.24-2.27) ----
    print(f"\n  [Final Fitting] Training multinomial logistic regression with selected {len(A_hat)} features (C={FINAL_C})...")
    t_final0 = time.time()
    X_train_sel = X_train_std[:, A_hat]
    X_test_sel = X_test_std[:, A_hat]
    final_clf = LogisticRegression(
        penalty="l2", C=FINAL_C, solver="lbfgs",
        max_iter=3000, multi_class="multinomial"
    )
    final_clf.fit(X_train_sel, y_train_full)
    t_final1 = time.time()
    final_fit_time = t_final1 - t_final0

    t_inf0 = time.time()
    test_pred = final_clf.predict(X_test_sel)
    t_inf1 = time.time()
    inference_time = t_inf1 - t_inf0
    inference_time_per_sample = inference_time / X_test.shape[0]

    train_pred = final_clf.predict(X_train_sel)
    train_acc = np.mean(train_pred == y_train_full)
    test_acc = np.mean(test_pred == y_test)

    total_time = time.time() - pipeline_t0
    hyperparam_search_time = t_ovr1 - t_ovr0

    df_final = len(A_hat) * (K - 1) + (K - 1)   # Multinomial model: (K-1) coefficient vectors + intercept
    total_params_full = p_col * (K - 1) + (K - 1)
    compression_ratio = df_final / total_params_full

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
    print(f"  ── General Metrics (Directly comparable with DAGFR/Spline) ──")
    print(f"    Train Accuracy              = {train_acc:.4f}")
    print(f"    Test Accuracy               = {test_acc:.4f}")
    print(f"    Train-Test Accuracy Gap     = {train_acc - test_acc:.4f}")
    print(f"    Hyperparameter Search Time  = {hyperparam_search_time:.1f} s "
          f"({K} OvR subproblems in parallel, each with Ridge-CV + BIC search along path)")
    print(f"    Final Model Fit Time        = {final_fit_time:.2f} s")
    print(f"    Total Training Time         = {total_time:.1f} s")
    print(f"    Inference Time (Full Test)  = {inference_time*1000:.2f} ms "
          f"({inference_time_per_sample*1e6:.2f} μs/sample)")
    print(f"    Degrees of Freedom (df)     = {df_final} / Max Params {total_params_full} "
          f"(Compression Ratio={compression_ratio:.3f})")
    print(f"\n  ── Adaptive Lasso Specifics (Reflecting OvR per-class sparsity + column-level selection) ──")
    print(f"    Final Feature Subset Size |Â|= {len(A_hat)} / {p_col} "
          f"(Aggregation Rule: {agg['rule_used']})")
    print(f"    Fully Excluded Blocks       = {n_fully_zero_groups}/{group_struct.n_groups}")
    print(f"    Partially Selected Blocks   = {n_partial_groups}/{group_struct.n_groups}  "
          f"(★Key structural difference from DAGFR: DAGFR does group-level all-or-nothing "
          f"selection, while Adaptive Lasso can select partial basis coefficient columns within a block)")
    print(f"    Fully Selected Blocks       = {n_full_groups}/{group_struct.n_groups}")
    print(f"    Selected Features per OvR   = "
          f"{[len(r['active_set']) for r in class_results]}")
    print(f"    Selected λ_c* per OvR       = "
          f"{[round(r['lambda_star'],5) for r in class_results]}")
    print(f"    Selected C_ridge* per OvR   = "
          f"{[round(r['C_ridge_star'],4) for r in class_results]}")

    print(f"\n  [Test Set Confusion Matrix] (Rows=True Class, Cols=Predicted Class, Class Codes={codes.tolist()})")
    print(f"  {'':>6s}" + "".join(f"pred={c:<6}" for c in codes))
    for i, row in enumerate(cm):
        print(f"  True={codes[i]:<3}" + "".join(f"{v:<11d}" for v in row))

    print(f"\n  [Per-class Precision/Recall/F1]")
    print("  " + report_str.replace("\n", "\n  "))
    print(f"{'-'*74}")

    return {
        "label": label, "p_col": p_col, "K": K,
        "n_active_features": len(A_hat), "aggregation_rule": agg["rule_used"],
        "n_fully_zero_groups": n_fully_zero_groups,
        "n_partial_groups": n_partial_groups, "n_full_groups": n_full_groups,
        "n_groups": group_struct.n_groups,
        "df_final": df_final, "total_params_full": total_params_full,
        "compression_ratio": compression_ratio,
        "train_acc": train_acc, "test_acc": test_acc,
        "hyperparam_search_time": hyperparam_search_time,
        "final_fit_time": final_fit_time, "total_train_time": total_time,
        "inference_time": inference_time,
        "inference_time_per_sample": inference_time_per_sample,
        "confusion_matrix": cm, "activity_codes": codes,
    }


if __name__ == "__main__":
    DESIGN_DIR = "/Users/augleovo/PycharmProjects/Application_New_副本/Experiment/design_matrices"

    result = run_adaptive_lasso(
        f"{DESIGN_DIR}/design_matrix_per_channel_Mk.npz", "Adaptive Lasso (Per-channel M_k)"
    )

    print(f"\n\n{'='*74}")
    print(f"  [For Chapter 5 Usage] Adaptive Lasso Summary of Final Metrics")
    print(f"{'='*74}")
    for k, v in result.items():
        if k == "confusion_matrix":
            continue
        print(f"  {k:<28s}: {v}")
    print(f"{'='*74}")