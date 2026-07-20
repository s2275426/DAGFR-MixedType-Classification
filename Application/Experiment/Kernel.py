import time
import numpy as np
import pandas as pd
from scipy import stats, signal
from scipy.optimize import minimize, minimize_scalar
from sklearn.metrics import confusion_matrix, classification_report
import warnings

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ============================================================
# Global constants aligned with preprocessing script (Do not modify)
# ============================================================
RAW_CSV_PATH = "/Users/augleovo/PycharmProjects/Application_New_副本/combined_devices_data.csv"
VALIDATION_NPZ_PATH = ("/Users/augleovo/PycharmProjects/Application_New_副本/"
                        "Experiment/design_matrices/design_matrix_per_channel_Mk.npz")

CHANNEL_COLUMNS = [
    "attitude.roll", "attitude.pitch", "attitude.yaw",
    "gravity.x", "gravity.y", "gravity.z",
    "rotationRate.x", "rotationRate.y", "rotationRate.z",
    "userAcceleration.x", "userAcceleration.y", "userAcceleration.z",
]
CHANNEL_NAMES = [c.replace(".", "_") for c in CHANNEL_COLUMNS]
N_CHANNELS = len(CHANNEL_COLUMNS)

STATIC_COVARIATE_COLUMNS = ["weight", "height", "age"]
N_TEST_SUBJECTS = 7
N_SPLIT_CANDIDATES = 5000
TRIM_SECONDS = 1.0
SAMPLING_RATE = 50.0
DT = 1.0 / SAMPLING_RATE

# ============================================================
# Kernel Method Hyperparameters
# ============================================================
N_KERNEL_FIT = 600              # Subsample size for weight optimization LOO phase (O(n²) full set infeasible)
N_TRAIN_EVAL_SUBSAMPLE = 3000    # Subsample size for training accuracy diagnostic
OMEGA_UPPER = 5.0
RIDGE_LAMBDA = 0.02
ZERO_THRESHOLD_RATIO = 0.05
REFERENCE_CHUNK_SIZE = 300       # Chunk size for final classification to manage memory usage

RAW_PREDICTOR_NAMES = CHANNEL_NAMES + STATIC_COVARIATE_COLUMNS + ["gender"]
RAW_PREDICTOR_TYPES = ["functional"] * N_CHANNELS + ["continuous"] * 3 + ["dummy"]
P_RAW = len(RAW_PREDICTOR_NAMES)


# ============================================================
# Deterministic functions reproducing exact window splitting
# ============================================================
def load_combined_data(csv_path):
    print(f"[Loading] Reading {csv_path} ...")
    df = pd.read_csv(csv_path)
    return df


def select_subject_split(df, n_test=N_TEST_SUBJECTS, n_candidates=N_SPLIT_CANDIDATES, seed=0):
    print(f"\n[Decision Point 1] Searching optimal split for {n_test} test subjects across {df['id'].nunique()} total subjects...")
    subj_info = df.groupby("id").agg(
        weight=("weight", "first"), height=("height", "first"),
        age=("age", "first"), gender=("gender", "first"),
    ).reset_index()
    all_ids = subj_info["id"].values
    male_ratio_all = (subj_info["gender"] == 1).mean()
    expected_male_test = n_test * male_ratio_all
    allowed_male_counts = set(range(int(np.floor(expected_male_test - 0.6)),
                                     int(np.ceil(expected_male_test + 0.6)) + 1))
    allowed_male_counts = {c for c in allowed_male_counts if 0 <= c <= n_test}

    rng = np.random.RandomState(seed)
    best_score, best_test_ids = np.inf, None
    n_tried, n_rejected = 0, 0

    train_full_w = subj_info["weight"].values
    train_full_h = subj_info["height"].values
    train_full_a = subj_info["age"].values

    while n_tried - n_rejected < n_candidates:
        n_tried += 1
        cand_test = rng.choice(all_ids, size=n_test, replace=False)
        cand_mask = subj_info["id"].isin(cand_test)
        n_male_test = (subj_info.loc[cand_mask, "gender"] == 1).sum()
        if n_male_test not in allowed_male_counts:
            n_rejected += 1
            continue

        train_mask = ~cand_mask
        score = 0.0
        for col, full_vals in [("weight", train_full_w), ("height", train_full_h), ("age", train_full_a)]:
            ks_stat, _ = stats.ks_2samp(subj_info.loc[train_mask, col], subj_info.loc[cand_mask, col])
            score += ks_stat
        if score < best_score:
            best_score = score
            best_test_ids = cand_test.copy()

    test_ids = sorted(best_test_ids.tolist())
    train_ids = sorted([i for i in all_ids if i not in test_ids])
    return train_ids, test_ids


def identify_static_dynamic_activities(df_train, trim_seconds=TRIM_SECONDS, fs=SAMPLING_RATE, ratio_threshold=5.0):
    trim_n = int(trim_seconds * fs)
    acc_cols = [c for c in CHANNEL_COLUMNS if c.startswith("userAcceleration")]
    variances_by_act = {}
    for (sid, act, trial), g in df_train.groupby(["id", "act", "trial"]):
        g = g.iloc[trim_n:-trim_n] if len(g) > 2 * trim_n else g
        if len(g) < 5: continue
        acc_mag = np.sqrt((g[acc_cols].values ** 2).sum(axis=1))
        variances_by_act.setdefault(act, []).append(np.var(acc_mag))
    act_median_var = {act: np.median(v) for act, v in variances_by_act.items()}
    sorted_acts = sorted(act_median_var.items(), key=lambda x: x[1])
    static_codes = [sorted_acts[0][0], sorted_acts[1][0]]
    dynamic_codes = [a for a, _ in sorted_acts[2:]]
    return static_codes, dynamic_codes


def analyze_dominant_frequency(df_train, dynamic_codes, fs=SAMPLING_RATE, trim_seconds=TRIM_SECONDS):
    trim_n = int(trim_seconds * fs)
    acc_cols = [c for c in CHANNEL_COLUMNS if c.startswith("userAcceleration")]
    dom_freq_by_act = {}
    for act in dynamic_codes:
        freqs_list = []
        sub = df_train[df_train["act"] == act]
        for (sid, trial), g in sub.groupby(["id", "trial"]):
            g = g.iloc[trim_n:-trim_n] if len(g) > 2 * trim_n else g
            if len(g) < 20: continue
            acc_mag = np.sqrt((g[acc_cols].values ** 2).sum(axis=1))
            acc_mag = signal.detrend(acc_mag, type="linear")
            f, pxx = signal.welch(acc_mag, fs=fs, nperseg=min(256, len(acc_mag)))
            dom_freq = f[np.argmax(pxx[1:]) + 1]
            freqs_list.append(dom_freq)
        dom_freq_by_act[act] = np.median(freqs_list)
    slowest_act = min(dom_freq_by_act, key=dom_freq_by_act.get)
    return dom_freq_by_act, dom_freq_by_act[slowest_act]


def compute_window_length_bounds(df_train, f_min, fs=SAMPLING_RATE, n_cycles=3, max_ratio=2, pct=5):
    L_min = int(np.ceil(n_cycles / f_min * fs))
    trial_lengths = df_train.groupby(["id", "act", "trial"]).size().values
    L_max = int(np.percentile(trial_lengths, pct) // max_ratio)
    return L_min, L_max


def _count_windows(df, L, step):
    total, min_count = 0, np.inf
    for (sid, act), g in df.groupby(["id", "act"]):
        g = g.sort_values("trial") if "trial" in g.columns else g
        n_group_windows = 0
        for trial, gt in g.groupby("trial"):
            n = len(gt)
            n_group_windows += max(0, (n - L) // step + 1)
        total += n_group_windows
        min_count = min(min_count, n_group_windows)
    return total, int(min_count) if min_count != np.inf else 0


def select_window_length_and_step(df_train, L_min, L_max, overlap_candidates=(0.0, 0.25, 0.5),
                                    step_multiples=(160, 192, 224, 256)):
    L_candidates = [L for L in step_multiples if L_min <= L <= L_max]
    best_score, best_L, best_overlap, best_step = -np.inf, None, None, None
    for L in L_candidates:
        for ov in overlap_candidates:
            step = int(L * (1 - ov))
            total_windows, min_per_group = _count_windows(df_train, L, step)
            score = min_per_group
            if score > best_score and total_windows > 0:
                best_score, best_L, best_overlap, best_step = score, L, ov, step
    return best_L, best_step


def compute_channel_standardization(df_train):
    means = {col: df_train[col].mean() for col in CHANNEL_COLUMNS}
    stds = {col: df_train[col].std() for col in CHANNEL_COLUMNS}
    return means, stds


def apply_channel_standardization(df, means, stds):
    df = df.copy()
    for col in CHANNEL_COLUMNS:
        df[col] = (df[col] - means[col]) / stds[col]
    return df


def compute_static_covariate_standardization(df_train, train_ids):
    subj_info = df_train[df_train["id"].isin(train_ids)].groupby("id").agg(
        weight=("weight", "first"), height=("height", "first"), age=("age", "first"),
    )
    return subj_info.mean().to_dict(), subj_info.std().to_dict()


def build_static_covariate_table(df, subject_ids, static_means, static_stds):
    table = {}
    for sid in subject_ids:
        row = df[df["id"] == sid].iloc[0]
        w = (row["weight"] - static_means["weight"]) / static_stds["weight"]
        h = (row["height"] - static_means["height"]) / static_stds["height"]
        a = (row["age"] - static_means["age"]) / static_stds["age"]
        g = row["gender"]
        table[sid] = np.array([w, h, a, g])
    return table


def make_windows(df, L, step, act_filter=None):
    windows = []
    for (sid, act, trial), g in df.groupby(["id", "act", "trial"]):
        if act_filter is not None and act not in act_filter: continue
        arr = g[CHANNEL_COLUMNS].values
        n = len(arr)
        for start in range(0, n - L + 1, step):
            windows.append({"data": arr[start:start + L], "act": act, "id": sid})
    return windows


# ============================================================
# Window list -> Raw curve arrays
# ============================================================
def windows_to_raw_arrays(windows, static_table):
    X_curves = np.stack([w["data"] for w in windows], axis=0)
    y = np.array([w["act"] for w in windows])
    ids = np.array([w["id"] for w in windows])
    static_feats = np.array([static_table[w["id"]] for w in windows])
    return X_curves, y, ids, static_feats


# ============================================================
# Normalization constants (computed on full training set raw curves)
# ============================================================
def compute_normalizing_constants_raw(X_train_curves, static_train, dt=DT):
    n = X_train_curves.shape[0]
    constants = np.zeros(P_RAW)
    for m in range(N_CHANNELS):
        channel = X_train_curves[:, :, m]
        mean_curve = channel.mean(axis=0)
        dev = channel - mean_curve
        sq_norm_i = dt * np.sum(dev ** 2, axis=1)
        constants[m] = np.sqrt(np.sum(sq_norm_i) / (n - 1))
    for j, static_col in enumerate([0, 1, 2, 3]):
        col = static_train[:, static_col]
        mu = col.mean()
        constants[N_CHANNELS + j] = np.sqrt(np.sum((col - mu) ** 2) / (n - 1))
    constants = np.maximum(constants, 1e-8)
    return constants


def pairwise_functional_distance_raw(A, B, dt=DT):
    sq_A = np.sum(A ** 2, axis=1)
    sq_B = np.sum(B ** 2, axis=1)
    cross = A @ B.T
    dist2 = sq_A[:, None] + sq_B[None, :] - 2 * cross
    dist2 = np.clip(dist2, 0, None)
    return np.sqrt(dt) * np.sqrt(dist2)


def pairwise_continuous_distance(a_col, b_col):
    return np.abs(a_col[:, None] - b_col[None, :])


def pairwise_dummy_distance(a_col, b_col):
    return (a_col[:, None] != b_col[None, :]).astype(float)


def compute_all_component_matrices_raw(X_ref_curves, static_ref, X_query_curves, static_query,
                                         norm_constants, dt=DT):
    mats = {}
    for m in range(N_CHANNELS):
        A = X_ref_curves[:, :, m]
        B = X_query_curves[:, :, m]
        mats[m] = pairwise_functional_distance_raw(A, B, dt) / norm_constants[m]
    mats[N_CHANNELS + 0] = pairwise_continuous_distance(static_ref[:, 0], static_query[:, 0]) / norm_constants[N_CHANNELS + 0]
    mats[N_CHANNELS + 1] = pairwise_continuous_distance(static_ref[:, 1], static_query[:, 1]) / norm_constants[N_CHANNELS + 1]
    mats[N_CHANNELS + 2] = pairwise_continuous_distance(static_ref[:, 2], static_query[:, 2]) / norm_constants[N_CHANNELS + 2]
    mats[N_CHANNELS + 3] = pairwise_dummy_distance(static_ref[:, 3], static_query[:, 3]) / norm_constants[N_CHANNELS + 3]
    return mats


def combine_composite_distance(component_mats, omega, n_groups=P_RAW):
    D = np.zeros_like(next(iter(component_mats.values())))
    for m in range(n_groups):
        D += omega[m] * component_mats[m]
    return D


# ============================================================
# Picard Kernel + LOO Brier Score (Weight Optimization Phase)
# ============================================================
def picard_kernel(D):
    return np.exp(-D)


def loo_brier_score(omega, component_mats, y_onehot, n_groups, ridge_lambda):
    D = combine_composite_distance(component_mats, omega, n_groups)
    Kmat = picard_kernel(D)
    np.fill_diagonal(Kmat, 0.0)
    row_sums = Kmat.sum(axis=1)
    row_sums = np.where(row_sums < 1e-12, 1e-12, row_sums)
    P_loo = (Kmat @ y_onehot) / row_sums[:, None]
    brier = np.sum((y_onehot - P_loo) ** 2)
    return brier + ridge_lambda * np.sum(omega ** 2)


def single_predictor_warm_start(component_mats, y_onehot, n_groups, ridge_lambda, omega_upper):
    omega0 = np.zeros(n_groups)
    for m in range(n_groups):
        def obj_1d(w_scalar):
            omega_vec = np.zeros(n_groups)
            omega_vec[m] = max(w_scalar, 0.0)
            return loo_brier_score(omega_vec, component_mats, y_onehot, n_groups, ridge_lambda)
        res = minimize_scalar(obj_1d, bounds=(0, omega_upper), method="bounded")
        omega0[m] = res.x
    return omega0


def optimize_weights(component_mats, y_onehot, n_groups, ridge_lambda, omega_upper):
    omega0 = single_predictor_warm_start(component_mats, y_onehot, n_groups, ridge_lambda, omega_upper)
    brier_at_warmstart = loo_brier_score(omega0, component_mats, y_onehot, n_groups, ridge_lambda)

    def obj(omega):
        return loo_brier_score(omega, component_mats, y_onehot, n_groups, ridge_lambda)

    bounds = [(0, omega_upper)] * n_groups
    res = minimize(obj, omega0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 200})

    return {
        "omega_star": res.x, "brier_star": res.fun,
        "omega_warmstart": omega0, "brier_warmstart": brier_at_warmstart,
        "converged": res.success, "n_iter": res.nit,
    }


# ============================================================
# Final Classification: Reference set set to full training set with chunking to avoid memory overflow
# ============================================================
def kernel_predict_proba_chunked(X_ref_curves, static_ref, y_ref_onehot,
                                   X_query_curves, static_query, norm_constants, omega,
                                   chunk_size=REFERENCE_CHUNK_SIZE):
    """
    Computes Eq 2.39 between reference set (X_ref_curves, etc.) and query set (X_query_curves, etc.).
    Chunks the query set to manage memory usage; mathematically identical to single-pass evaluation.
    """
    n_query = X_query_curves.shape[0]
    K = y_ref_onehot.shape[1]
    P_out = np.zeros((n_query, K))

    for start in range(0, n_query, chunk_size):
        end = min(start + chunk_size, n_query)
        X_chunk = X_query_curves[start:end]
        static_chunk = static_query[start:end]

        comp_mats = compute_all_component_matrices_raw(
            X_ref_curves, static_ref, X_chunk, static_chunk, norm_constants
        )   # Each component matrix shape: (n_ref, chunk_size)
        D = combine_composite_distance(comp_mats, omega, P_RAW)   # (n_ref, chunk_size)
        Kmat = picard_kernel(D)
        row_sums = Kmat.sum(axis=0)                                # Sum over reference set -> (chunk_size,)
        row_sums = np.where(row_sums < 1e-12, 1e-12, row_sums)
        P_out[start:end] = (Kmat.T @ y_ref_onehot) / row_sums[:, None]

    return P_out


def stratified_subsample(y, size, K, random_state=RANDOM_STATE):
    rng = np.random.RandomState(random_state)
    idx_all = []
    per_class = max(size // K, 1)
    for c in range(K):
        idx_c = np.where(y == c)[0]
        chosen = rng.choice(idx_c, size=min(per_class, len(idx_c)), replace=False)
        idx_all.append(chosen)
    idx_all = np.concatenate(idx_all)
    rng.shuffle(idx_all)
    return idx_all[:size]


# ============================================================
# Main Execution Pipeline
# ============================================================
def run_weighted_kernel_raw():
    label = "Weighted Kernel (Raw Curves, Full Training Reference Set, Exact d_fun)"
    print(f"\n{'='*74}")
    print(f"  {label}")
    print(f"{'='*74}")

    pipeline_t0 = time.time()

    df = load_combined_data(RAW_CSV_PATH)
    train_ids, test_ids = select_subject_split(df, N_TEST_SUBJECTS)

    df_train_raw = df[df["id"].isin(train_ids)].reset_index(drop=True)
    df_test_raw = df[df["id"].isin(test_ids)].reset_index(drop=True)

    static_codes, dynamic_codes = identify_static_dynamic_activities(df_train_raw)
    _, f_min = analyze_dominant_frequency(df_train_raw, dynamic_codes)
    L_min, L_max = compute_window_length_bounds(df_train_raw, f_min)
    L_star, train_step_star = select_window_length_and_step(df_train_raw, L_min, L_max)
    print(f"\n  Reproduced Window Parameters: L_star={L_star}, train_step_star={train_step_star}")

    channel_means, channel_stds = compute_channel_standardization(df_train_raw)
    df_train_std = apply_channel_standardization(df_train_raw, channel_means, channel_stds)
    df_test_std = apply_channel_standardization(df_test_raw, channel_means, channel_stds)

    static_means, static_stds = compute_static_covariate_standardization(df, train_ids)
    static_table_train = build_static_covariate_table(df, train_ids, static_means, static_stds)
    static_table_test = build_static_covariate_table(df, test_ids, static_means, static_stds)
    static_table_all = {**static_table_train, **static_table_test}

    train_windows = make_windows(df_train_std, L_star, train_step_star)
    test_windows = make_windows(df_test_std, L_star, L_star)
    print(f"  Window Statistics: Train Set={len(train_windows)} windows, Test Set={len(test_windows)} windows")

    X_train_curves, y_train, id_train, static_train = windows_to_raw_arrays(train_windows, static_table_all)
    X_test_curves, y_test, id_test, static_test = windows_to_raw_arrays(test_windows, static_table_all)
    K = len(np.unique(np.concatenate([y_train, y_test])))
    print(f"  Raw Curve Arrays: X_train_curves.shape={X_train_curves.shape}, "
          f"X_test_curves.shape={X_test_curves.shape}, K={K}")
    print(f"  Δt = 1/{SAMPLING_RATE} = {DT:.4f}s")

    try:
        data_check = np.load(VALIDATION_NPZ_PATH, allow_pickle=True)
        y_train_npz = data_check["y_train"]
        y_test_npz = data_check["y_test"]
        if len(y_train) == len(y_train_npz) and len(y_test) == len(y_test_npz):
            train_match_ratio = np.mean(y_train == y_train_npz)
            test_match_ratio = np.mean(y_test == y_test_npz)
            print(f"\n  [Alignment Verification] Train match ratio={train_match_ratio:.6f}, Test match ratio={test_match_ratio:.6f}")
    except Exception as e:
        print(f"\n  [Alignment Verification] Skipped ({e})")

    norm_constants = compute_normalizing_constants_raw(X_train_curves, static_train)
    print(f"\n  Normalization Constants c_j (Eq 2.35):")
    for m in range(P_RAW):
        print(f"    {RAW_PREDICTOR_NAMES[m]:<22s}: c_j={norm_constants[m]:.4f}")

    # ---- Weight Optimization: Subsample with n_kernel_fit=600 ----
    print(f"\n  [Weight Optimization] Extracting stratified subsample with n_kernel_fit={N_KERNEL_FIT} for LOO Brier optimization")
    fit_idx = stratified_subsample(y_train, N_KERNEL_FIT, K)
    X_fit_curves = X_train_curves[fit_idx]
    static_fit = static_train[fit_idx]
    y_fit_onehot = np.eye(K)[y_train[fit_idx]]

    component_mats_fit = compute_all_component_matrices_raw(
        X_fit_curves, static_fit, X_fit_curves, static_fit, norm_constants
    )

    t_opt0 = time.time()
    opt_result = optimize_weights(component_mats_fit, y_fit_onehot, P_RAW, RIDGE_LAMBDA, OMEGA_UPPER)
    weight_opt_time = time.time() - t_opt0
    omega_star = opt_result["omega_star"]
    print(f"  Optimized Brier (+ridge)={opt_result['brier_star']:.2f}, Elapsed time={weight_opt_time:.1f}s")

    n_soft_zero = int(np.sum(omega_star < ZERO_THRESHOLD_RATIO * OMEGA_UPPER))
    d_effective = P_RAW - n_soft_zero
    print(f"\n  Soft-zero predictors count = {n_soft_zero}/{P_RAW}, d_effective={d_effective}")

    # ============================================================
    # Final Classification: Reference Set = Full Training Set
    # ============================================================
    print(f"\n  [Final Classification] Reference set = Full training set n_train={len(y_train)} "
          f"(chunk_size={REFERENCE_CHUNK_SIZE} used to manage memory during test set evaluation)")

    y_train_onehot_full = np.eye(K)[y_train]

    t_inf0 = time.time()
    P_test = kernel_predict_proba_chunked(
        X_train_curves, static_train, y_train_onehot_full,
        X_test_curves, static_test, norm_constants, omega_star,
        chunk_size=REFERENCE_CHUNK_SIZE
    )
    test_pred = np.argmax(P_test, axis=1)
    inference_time = time.time() - t_inf0
    inference_time_per_sample = inference_time / X_test_curves.shape[0]
    test_acc = np.mean(test_pred == y_test)

    # ---- Training Accuracy Diagnostic ----
    train_eval_idx = stratified_subsample(y_train, N_TRAIN_EVAL_SUBSAMPLE, K, random_state=RANDOM_STATE + 2)
    ref_for_train_eval_idx = stratified_subsample(y_train, N_TRAIN_EVAL_SUBSAMPLE, K, random_state=RANDOM_STATE + 3)
    X_ref_diag = X_train_curves[ref_for_train_eval_idx]
    static_ref_diag = static_train[ref_for_train_eval_idx]
    y_ref_diag_onehot = np.eye(K)[y_train[ref_for_train_eval_idx]]

    P_train = kernel_predict_proba_chunked(
        X_ref_diag, static_ref_diag, y_ref_diag_onehot,
        X_train_curves[train_eval_idx], static_train[train_eval_idx],
        norm_constants, omega_star, chunk_size=REFERENCE_CHUNK_SIZE
    )
    train_pred = np.argmax(P_train, axis=1)
    train_acc = np.mean(train_pred == y_train[train_eval_idx])

    total_time = time.time() - pipeline_t0

    cm = confusion_matrix(y_test, test_pred)
    report_str = classification_report(
        y_test, test_pred, target_names=[f"act={c}" for c in np.unique(y_test)], digits=4
    )

    print(f"\n{'-'*74}")
    print(f"  [Chapter 5 Comparison Metrics Summary - {label}]")
    print(f"{'-'*74}")
    print(f"    Training Set Accuracy (Subsample Diagnostic, n={N_TRAIN_EVAL_SUBSAMPLE}, Non-strict LOO) = {train_acc:.4f}")
    print(f"    Test Set Accuracy (Reference Set = Full Train Set {len(y_train)}) = {test_acc:.4f}")
    print(f"    Weight Optimization Time   = {weight_opt_time:.1f} s (based on n_kernel_fit={N_KERNEL_FIT} subsample)")
    print(f"    Inference Time (Full Test)  = {inference_time*1000:.2f} ms "
          f"({inference_time_per_sample*1e6:.2f} μs/sample, reference set size={len(y_train)})")
    print(f"    Total Pipeline Time         = {total_time:.1f} s")
    print(f"    Final ω Weight Vector       = {np.round(omega_star, 4).tolist()}")
    print(f"    Soft-zero Predictors Count  = {n_soft_zero}/{P_RAW}, d_effective={d_effective}")

    print(f"\n  [Test Set Confusion Matrix]")
    codes = np.unique(y_test)
    print(f"  {'':>6s}" + "".join(f"pred={c:<6}" for c in codes))
    for i, row in enumerate(cm):
        print(f"  true={codes[i]:<3}" + "".join(f"{v:<11d}" for v in row))
    print(f"\n  [Per-class Precision / Recall / F1-Score]")
    print("  " + report_str.replace("\n", "\n  "))
    print(f"{'-'*74}")

    return {
        "label": label, "p_raw": P_RAW, "K": K, "L_star": L_star,
        "omega_star": omega_star.tolist(),
        "n_soft_zero": n_soft_zero, "d_effective": d_effective,
        "train_acc": train_acc, "test_acc": test_acc,
        "weight_opt_time": weight_opt_time, "total_train_time": total_time,
        "inference_time": inference_time,
        "inference_time_per_sample": inference_time_per_sample,
        "brier_star": opt_result["brier_star"],
        "reference_set_size": len(y_train),
        "confusion_matrix": cm,
    }


if __name__ == "__main__":
    result = run_weighted_kernel_raw()
    print(f"\n\n{'='*74}")
    print(f"  [For Chapter 5 Usage] Weighted Kernel (Full Reference Set) Final Metrics Overview")
    print(f"{'='*74}")
    for k, v in result.items():
        if k == "confusion_matrix":
            continue
        print(f"  {k:<28s}: {v}")
    print(f"{'='*74}")