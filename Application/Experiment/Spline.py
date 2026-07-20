import time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ============================================================
# Fixed Hyperparameter from Paper (Section 2.4.4, Algorithm 2)
# ============================================================
FIXED_C = 1.0


def load_npz_dataset(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    X_train, y_train_raw = data["X_train"], data["y_train"]
    X_test, y_test_raw = data["X_test"], data["y_test"]
    group_names = data["group_names"]
    group_boundaries = data["group_boundaries"]

    all_codes = np.unique(np.concatenate([y_train_raw, y_test_raw]))
    code_to_idx = {c: i for i, c in enumerate(all_codes)}
    y_train = np.array([code_to_idx[c] for c in y_train_raw])
    y_test = np.array([code_to_idx[c] for c in y_test_raw])

    return X_train, y_train, X_test, y_test, all_codes, group_names, group_boundaries


def run_spline_pipeline(npz_path, label):
    print(f"\n{'='*74}")
    print(f"  Algorithm 2 (Spline Pipeline) Running: {label}")
    print(f"  Data Source: {npz_path}")
    print(f"{'='*74}")

    pipeline_t0 = time.time()

    X_train, y_train, X_test, y_test, activity_codes, group_names, group_boundaries = \
        load_npz_dataset(npz_path)
    n_train, p_col = X_train.shape
    K = len(activity_codes)
    print(f"  n_train={n_train}, n_test={X_test.shape[0]}, p_col={p_col}, K={K}")
    print(f"  Fixed Regularization Strength C = {FIXED_C} (Specified value in Section 2.4.4, no tuning)")

    # ---- Step 3: Multinomial Logistic Regression, L2 penalty, C=1.0, lbfgs, multinomial ----
    clf = LogisticRegression(
        penalty="l2",
        C=FIXED_C,
        solver="lbfgs",
        max_iter=5000,
        tol=1e-2,
        multi_class="multinomial",
    )

    t_fit_start = time.time()
    clf.fit(X_train, y_train)
    t_fit_end = time.time()
    fit_time = t_fit_end - t_fit_start

    t_inf_start = time.time()
    test_pred = clf.predict(X_test)
    t_inf_end = time.time()
    inference_time = t_inf_end - t_inf_start
    inference_time_per_sample = inference_time / X_test.shape[0]

    train_pred = clf.predict(X_train)
    train_acc = np.mean(train_pred == y_train)
    test_acc = np.mean(test_pred == y_test)

    total_pipeline_time = time.time() - pipeline_t0

    # ---- Parameter Count Statistics: Dense model, no structure selection, df = parameter upper bound, compression ratio strictly 1.0 ----
    # sklearn's coef_ contains coefficients for K classes (not K-1 reference class format); parameters counted by this standard
    df_dense = clf.coef_.size + clf.intercept_.size
    total_params_full = df_dense
    compression_ratio = 1.0   # Dense model has no compression; truthfully marked as 1.0

    print(f"\n  [Step 3 Results] Training Accuracy = {train_acc:.4f}   Test Accuracy = {test_acc:.4f}")
    print(f"  Train-Test Accuracy Gap = {train_acc - test_acc:.4f} "
          f"({'Gap is very small, indicating no severe overfitting under L2 penalty with C=1.0' if abs(train_acc-test_acc) < 0.08 else 'Gap is relatively large, pay attention to potential overfitting'})")
    print(f"  Fitting Time = {fit_time:.2f} s")
    print(f"  Inference Time (Full Test Set) = {inference_time*1000:.2f} ms "
          f"({inference_time_per_sample*1e6:.2f} μs/sample)")
    print(f"  Number of Parameters (Degrees of Freedom) df = {df_dense} (Dense model, no group sparsity/fusion selection mechanism)")

    # ---- Coefficient Norm Diagnostic: Cross-validation with DAGFR zero-group determination ----
    coef = clf.coef_
    print(f"\n  [Diagnostic] Coefficient norms for each functional block under dense L2 model (for cross-validation with DAGFR zero-group determination):")
    for m, (s, e) in enumerate(group_boundaries):
        block_norm = np.linalg.norm(coef[:, int(s):int(e)], ord="fro")
        print(f"    {str(group_names[m]):<22s}: ‖β‖_F = {block_norm:.4f}")

    # ---- Confusion Matrix / Classification Report ----
    cm = confusion_matrix(y_test, test_pred)
    print(f"\n  [Test Set Confusion Matrix] (Rows = True Classes, Columns = Predicted Classes, Class Order by Encoding = {activity_codes.tolist()})")
    print(f"  {'':>6s}" + "".join(f"pred={c:<6}" for c in activity_codes))
    for i, row in enumerate(cm):
        print(f"  true={activity_codes[i]:<3}" + "".join(f"{v:<11d}" for v in row))

    report_str = classification_report(
        y_test, test_pred, target_names=[f"act={c}" for c in activity_codes], digits=4
    )
    print(f"\n  [Per-class Precision / Recall / F1-Score]")
    print("  " + report_str.replace("\n", "\n  "))

    # ============================================================
    # Chapter 5 Complete Metrics Output (Aligned with dagfr_final_mk.py for side-by-side comparison)
    # ============================================================
    print(f"\n{'-'*74}")
    print(f"  [Chapter 5 Comparison Metrics Summary - {label}]")
    print(f"{'-'*74}")
    print(f"  ── General Level (Directly comparable with DAGFR) ──")
    print(f"    Training Set Accuracy  = {train_acc:.4f}")
    print(f"    Test Set Accuracy      = {test_acc:.4f}")
    print(f"    Train-Test Acc Gap     = {train_acc - test_acc:.4f}")
    print(f"    Total Training Time    = {total_pipeline_time:.2f} s "
          f"(Single lbfgs fit, no hyperparameter search — mandatory prerequisite note when comparing training time with DAGFR)")
    print(f"    Inference Time (Full)  = {inference_time*1000:.2f} ms "
          f"({inference_time_per_sample*1e6:.2f} μs/sample)")
    print(f"    Degrees of Freedom df  = {df_dense} / Upper Bound {total_params_full} "
          f"(Compression Ratio = {compression_ratio:.3f}, i.e., No Compression)")
    print(f"\n  ── DAGFR-Specific Layer (Spline is dense model; marked N/A as applicable) ──")
    print(f"    λ_P* / λ_F*            = N/A (Spline has no group sparsity/fusion penalty mechanism)")
    print(f"    Zero Groups / Total    = 0/{len(group_boundaries)} "
          f"(Dense model performs no variable selection; all functional block coefficients are non-zero, "
          f"even if block ‖β‖_F is very small as shown in diagnostics above)")
    print(f"    Effective Feature Cols = {p_col} / {p_col} "
          f"(Cannot yield deployment conclusions like 'which sensor channels can be saved' as DAGFR does)")
    print(f"    Fusion Compressed df   = N/A (No cross-class parameter binding mechanism)")
    print(f"{'-'*74}")

    return {
        "label": label, "p_col": p_col,
        "df_dense": df_dense, "total_params_full": total_params_full,
        "compression_ratio": compression_ratio,
        "train_acc": train_acc, "test_acc": test_acc,
        "fit_time": fit_time, "total_pipeline_time": total_pipeline_time,
        "inference_time": inference_time,
        "inference_time_per_sample": inference_time_per_sample,
        "confusion_matrix": cm, "activity_codes": activity_codes,
    }


if __name__ == "__main__":
    DESIGN_DIR = "/Users/augleovo/PycharmProjects/Application_New_副本/Experiment/design_matrices"

    # Per final decision, run only per-channel M_k path
    result = run_spline_pipeline(
        f"{DESIGN_DIR}/design_matrix_per_channel_Mk.npz", "Spline Baseline (Per-channel M_k)"
    )

    print(f"\n\n{'='*74}")
    print(f"  [For Chapter 5 Usage] Spline Baseline Final Metrics Overview")
    print(f"{'='*74}")
    for k, v in result.items():
        if k in ("confusion_matrix",):
            continue
        print(f"  {k:<28s}: {v}")
    print(f"{'='*74}")

    print(f"""
  Interpretation Rules (Reference for side-by-side comparison with dagfr_final_mk.py output):
  - If the test accuracy of the Spline baseline is significantly higher than DAGFR (e.g., >5 percentage points higher),
    it indicates that DAGFR's λ_P*/λ_F* may have eliminated some genuinely discriminative information.
  - If test accuracies are close (gap within 2-3 percentage points), but DAGFR's df is far smaller
    than Spline's {result['df_dense']} and DAGFR's "Effective Feature Columns" count is significantly lower than {result['p_col']},
    this demonstrates the core methodology advantage of DAGFR: trading far fewer degrees of freedom and far fewer
    physical measurement channels for practically equivalent predictive precision — expected successful case.
""")