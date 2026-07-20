import os
import json
import numpy as np
import pandas as pd
from scipy import stats, signal, linalg
from scipy.interpolate import BSpline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed

# ============================================================
# 全局常量 (保持一致)
# ============================================================
RAW_CSV_PATH = "/Users/augleovo/PycharmProjects/Application_New_副本/combined_devices_data.csv"
OUTPUT_DIR = "/Users/augleovo/PycharmProjects/Application_New_副本/Experiment/design_matrices"

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
SPLINE_DEGREE = 3           
DIFF_ORDER = 2              
TRIM_SECONDS = 1.0          

M_CANDIDATES = [8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 72, 80, 84]
N_PROXY_REPEATS = 15
N_VAL_SUBJECTS = 5
PROXY_C = 1.0
PROXY_SEED = 42
SAMPLING_RATE = 50.0        

# ============================================================
# 向量化提速核心：移除 lambda 循环，改用 NumPy 广播
# ============================================================
def gcv_select_lambda_vectorized(y, B, T, V, S, lambda_grid):
    """
    通过完全向量化（Vectorized）消除对 lambda_grid 的 Python 循环
    y: (L,) 或 (L, N_CHANNELS)
    """
    if y.ndim == 1:
        y = y_arr = y[:, np.newaxis]
    else:
        y_arr = y
        
    # Bty 形状: (M, N_CHANNELS)
    Bty = B.T @ y_arr
    # VT_Bty 形状: (M, N_CHANNELS)
    VT_Bty = V.T @ Bty
    
    # S 形状: (M,) -> 扩展为 (30, M, 1) 用于广播
    # lambda_grid 形状: (30,) -> 扩展为 (30, 1, 1)
    S_exp = S[np.newaxis, :, np.newaxis]
    lam_exp = lambda_grid[:, np.newaxis, np.newaxis]
    
    # denom_diag 形状: (30, M, 1)
    denom_diag = 1.0 + lam_exp * S_exp
    
    # trace_H 形状: (30,)
    trace_H = np.sum(1.0 / denom_diag[:, :, 0], axis=1)
    denom_gcv = T - trace_H
    
    # 过滤无效的 denom
    valid_mask = denom_gcv > 0
    if not np.any(valid_mask):
        return lambda_grid[0], np.zeros((B.shape[1], y_arr.shape[1]))
        
    # 核心计算：利用广播一次性求出所有 lambda 下的 c
    # (30, M, N_CHANNELS)
    c_all_lam = V[np.newaxis, :, :] @ (VT_Bty[np.newaxis, :, :] / denom_diag)
    
    # 计算所有 lambda 下的 y_hat 和 gcv 分数
    # y_hat 形状: (30, L, N_CHANNELS)
    y_hat_all = B[np.newaxis, :, :] @ c_all_lam
    
    # 针对每个通道，找出使其 GCV 最小的那个 lambda 索引
    # residual_sq 形状: (30, N_CHANNELS)
    residual_sq = np.sum((y_arr[np.newaxis, :, :] - y_hat_all) ** 2, axis=1)
    
    # gcv_matrix 形状: (30, N_CHANNELS)
    gcv_matrix = (residual_sq / (denom_gcv[:, np.newaxis] ** 2)) * T
    # 对不合法的位置赋予无穷大
    gcv_matrix[~valid_mask, :] = np.inf
    
    best_lam_indices = np.argmin(gcv_matrix, axis=0)
    
    # 提取最终每个通道的最佳 c
    final_coeffs = np.zeros((B.shape[1], y_arr.shape[1]))
    for ch in range(y_arr.shape[1]):
        final_coeffs[:, ch] = c_all_lam[best_lam_indices[ch], :, ch]
        
    return final_coeffs

# 单个窗口的处理函数（抽离出来以便进行真实的多进程并行）
def _project_single_window_worker(w_data, bases_precomputed, lambda_grid, L):
    window_coeffs = {}
    for M, (B, V, S) in bases_precomputed.items():
        # 一次性把 12 个通道扔进去向量化计算
        coeffs = gcv_select_lambda_vectorized(w_data, B, L, V, S, lambda_grid)
        window_coeffs[M] = coeffs.T  # 转置回 (N_CHANNELS, M)
    return window_coeffs

# ============================================================
# 核心优化 1：更改为真实的多进程并行 backend="multiprocessing"
# ============================================================
def precompute_all_projections(windows, L, M_candidates, n_jobs=-1):
    print(f"\n[预计算] 正在高效多进程并行预计算 {len(windows)} 个窗口的 P-spline 投影系数...")
    
    lambda_grid = np.logspace(-4, 4, 30)
    bases_precomputed = {}
    
    # 移出循环：提前在主进程中完成矩阵分解，避免子进程重复计算
    for M in M_candidates:
        try:
            B = build_bspline_basis(L, M)
            D2 = build_diff_matrix(M)
            BtB = B.T @ B
            DtD = D2.T @ D2
            try:
                S, V = linalg.eigh(DtD, BtB + 1e-10 * np.eye(len(BtB)))
            except (linalg.LinAlgError, ValueError):
                S, V = linalg.eigh(DtD + 1e-8 * np.eye(len(DtD)), BtB + 1e-6 * np.eye(len(BtB)))
            bases_precomputed[M] = (B, V, S)
        except ValueError:
            continue

    # 关键修改：改用 multiprocessing 后，纯 Python 循环能真正利用多核 CPU
    results = Parallel(n_jobs=n_jobs, backend="multiprocessing")(
        delayed(_project_single_window_worker)(w["data"], bases_precomputed, lambda_grid, L) 
        for w in windows
    )
    
    precomputed = {}
    for M in bases_precomputed.keys():
        precomputed[M] = np.array([res[M] for res in results]) 
        
    print("  预计算完成！")
    return precomputed


# ============================================================
# 主流程中涉及数据预处理的辅助函数 (保持不变)
# ============================================================
def select_subject_split(df, n_test=N_TEST_SUBJECTS, n_candidates=N_SPLIT_CANDIDATES, seed=0):
    print(f"\n[决策点①] 在 {df['id'].nunique()} 个subject中搜索 {n_test} 人做测试集的最优划分...")
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
    return train_ids, test_ids, best_score

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

def select_window_length_and_step(df_train, L_min, L_max, overlap_candidates=(0.0, 0.25, 0.5), step_multiples=(160, 192, 224, 256)):
    L_candidates = [L for L in step_multiples if L_min <= L <= L_max]
    best_score, best_L, best_overlap, best_step = -np.inf, None, None, None
    for L in L_candidates:
        for ov in overlap_candidates:
            step = int(L * (1 - ov))
            total_windows, min_per_group = _count_windows(df_train, L, step)
            score = min_per_group
            if score > best_score and total_windows > 0:
                best_score, best_L, best_overlap, best_step = score, L, ov, step
    return best_L, best_step, best_overlap  

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

def load_combined_data(csv_path):
    print(f"[加载] 读取 {csv_path} ...")
    return pd.read_csv(csv_path)

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

def build_bspline_basis(L, M, degree=SPLINE_DEGREE):
    n0 = M - degree
    if n0 < 1: raise ValueError(f"M={M} 太小")
    x_min, x_max = 0.0, L - 1
    interior_knots = np.linspace(x_min, x_max, n0 + 1)
    step = interior_knots[1] - interior_knots[0]
    knots = np.concatenate([
        [x_min - step * k for k in range(degree, 0, -1)],
        interior_knots,
        [x_max + step * k for k in range(1, degree + 1)],
    ])
    t_eval = np.arange(L)
    B = np.zeros((L, M))
    for j in range(M):
        coeff = np.zeros(M)
        coeff[j] = 1.0
        spline = BSpline(knots, coeff, degree, extrapolate=False)
        B[:, j] = np.nan_to_num(spline(t_eval), nan=0.0)
    return B

def build_diff_matrix(M, order=DIFF_ORDER):
    D = np.eye(M)
    for _ in range(order): D = np.diff(D, axis=0)
    return D

# ============================================================
# 优化 2：基于缓存的快速 M 选择 (添加 LogisticRegression 的 n_jobs=-1 并行)
# ============================================================
def select_global_M_fast(train_windows, precomputed, static_table, train_subject_ids,
                        M_candidates=M_CANDIDATES, n_repeats=N_PROXY_REPEATS,
                        n_val_subjects=N_VAL_SUBJECTS, seed=PROXY_SEED):
    print(f"\n{'='*70}\n[路径A-快速] 全局M选择 (直接读取预计算缓存)\n{'='*70}")
    
    results = {}
    y_all = np.array([w["act"] for w in train_windows])
    id_all = np.array([w["id"] for w in train_windows])
    static_feats = np.array([static_table[w["id"]] for w in train_windows])
    
    rng = np.random.RandomState(seed)
    cv_splits = []
    for _ in range(n_repeats):
        val_subj = rng.choice(train_subject_ids, size=n_val_subjects, replace=False)
        cv_splits.append(np.isin(id_all, val_subj))

    for M in M_candidates:
        if M not in precomputed: continue
        
        coeffs_M = precomputed[M]  
        feat_spline = coeffs_M.reshape(len(train_windows), -1)
        X_all = np.hstack([feat_spline, static_feats])

        accs = []
        for val_mask in cv_splits:
            tr_mask = ~val_mask
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_all[tr_mask])
            X_val = scaler.transform(X_all[val_mask])

            # 优化点：多分类 LogisticRegression 拟合较慢，开启 n_jobs=-1 利用多核
            clf = LogisticRegression(penalty="l2", C=PROXY_C, max_iter=1000, solver="lbfgs", n_jobs=-1)
            clf.fit(X_tr, y_all[tr_mask])
            accs.append(clf.score(X_val, y_all[val_mask]))

        results[M] = accs
        print(f"  M={M:<4d} 完成, 平均验证准确率 = {np.mean(accs):.4f}")

    M_star = _apply_one_se_rule(results, M_candidates)
    print(f"\n  → [路径A] 最终选定全局 M* = {M_star}")
    return M_star, results


def select_per_channel_M_fast(train_windows, precomputed, static_table, train_subject_ids,
                             M_candidates=M_CANDIDATES, n_repeats=N_PROXY_REPEATS,
                             n_val_subjects=N_VAL_SUBJECTS, seed=PROXY_SEED):
    print(f"\n{'='*70}\n[路径B-快速] 逐通道独立M_k选择 (直接读取预计算缓存)\n{'='*70}")
    
    M_k_dict = {}
    all_results = {}
    y_all = np.array([w["act"] for w in train_windows])
    id_all = np.array([w["id"] for w in train_windows])
    static_feats = np.array([static_table[w["id"]] for w in train_windows])
    
    rng = np.random.RandomState(seed)
    cv_splits = []
    for _ in range(n_repeats):
        val_subj = rng.choice(train_subject_ids, size=n_val_subjects, replace=False)
        cv_splits.append(np.isin(id_all, val_subj))

    for ch_idx, ch_name in enumerate(CHANNEL_NAMES):
        print(f"\n  --- 通道: {ch_name} ---")
        results = {}
        for M in M_candidates:
            if M not in precomputed: continue
            
            c_single_channel = precomputed[M][:, ch_idx, :] 
            X_all = np.hstack([c_single_channel, static_feats])

            accs = []
            for val_mask in cv_splits:
                tr_mask = ~val_mask
                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X_all[tr_mask])
                X_val = scaler.transform(X_all[val_mask])

                # 优化点：多分类 LogisticRegression 拟合较慢，开启 n_jobs=-1 利用多核
                clf = LogisticRegression(penalty="l2", C=PROXY_C, max_iter=1000, solver="lbfgs", n_jobs=-1)
                clf.fit(X_tr, y_all[tr_mask])
                accs.append(clf.score(X_val, y_all[val_mask]))

            results[M] = accs
            print(f"    M={M:<4d} 平均验证准确率 = {np.mean(accs):.4f}")

        M_k = _apply_one_se_rule(results, M_candidates)
        M_k_dict[ch_name] = M_k
        all_results[ch_name] = results
        print(f"  → [{ch_name}] 选定 M_k = {M_k}")

    return M_k_dict, all_results


def _apply_one_se_rule(results, candidates):
    valid_candidates = [c for c in candidates if c in results]
    means = np.array([np.mean(results[c]) for c in valid_candidates])
    ses = np.array([np.std(results[c], ddof=1) / np.sqrt(len(results[c])) for c in valid_candidates])
    best_idx = int(np.argmax(means))
    threshold = means[best_idx] - ses[best_idx]
    eligible = [c for c, m in zip(valid_candidates, means) if m >= threshold]
    return min(eligible)


# ============================================================
# 利用预计算的特征快速构建 Design Matrix 
# ============================================================
def build_design_matrix_global_M_fast(windows, precomputed, M, static_table):
    coeffs_M = precomputed[M]  
    feat_spline = coeffs_M.reshape(len(windows), -1)
    static_feats = np.array([static_table[w["id"]] for w in windows])
    
    X = np.hstack([feat_spline, static_feats])
    y = np.array([w["act"] for w in windows])
    ids = np.array([w["id"] for w in windows])

    group_boundaries = []
    group_names = []
    ptr = 0
    for ch_name in CHANNEL_NAMES:
        group_boundaries.append((ptr, ptr + M))
        group_names.append(ch_name)
        ptr += M
    for static_name in STATIC_COVARIATE_COLUMNS + ["gender"]:
        group_boundaries.append((ptr, ptr + 1))
        group_names.append(static_name)
        ptr += 1

    return X, y, ids, np.array(group_boundaries), np.array(group_names, dtype=object)


def build_design_matrix_per_channel_Mk_fast(windows, precomputed_dict, M_k_dict, static_table):
    feats_list = []
    for ch_idx, ch_name in enumerate(CHANNEL_NAMES):
        M_k = M_k_dict[ch_name]
        feats_list.append(precomputed_dict[M_k][:, ch_idx, :]) 
        
    feat_spline = np.hstack(feats_list)
    static_feats = np.array([static_table[w["id"]] for w in windows])
    
    X = np.hstack([feat_spline, static_feats])
    y = np.array([w["act"] for w in windows])
    ids = np.array([w["id"] for w in windows])

    group_boundaries = []
    group_names = []
    ptr = 0
    for ch_name in CHANNEL_NAMES:
        M_k = M_k_dict[ch_name]
        group_boundaries.append((ptr, ptr + M_k))
        group_names.append(ch_name)
        ptr += M_k
    for static_name in STATIC_COVARIATE_COLUMNS + ["gender"]:
        group_boundaries.append((ptr, ptr + 1))
        group_names.append(static_name)
        ptr += 1

    return X, y, ids, np.array(group_boundaries), np.array(group_names, dtype=object)


# ============================================================
# 主流程
# ============================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = load_combined_data(RAW_CSV_PATH)
    train_ids, test_ids, best_score = select_subject_split(df, N_TEST_SUBJECTS)

    print(f"\n" + "="*50)
    print(f"【受试者划分详细结果】")
    print(f"  - 测试集 7 个受试者 ID : {test_ids}")
    print(f"  - 训练集 {len(train_ids)} 个受试者 ID: {train_ids}")
    print(f"  - 最小两样本 KS 统计量之和 (best_score): {best_score:.4f}")
    print("="*50 + "\n")

    df_train_raw = df[df["id"].isin(train_ids)].reset_index(drop=True)
    df_test_raw = df[df["id"].isin(test_ids)].reset_index(drop=True)

    static_codes, dynamic_codes = identify_static_dynamic_activities(df_train_raw)
    
    print(f"\n" + "="*50)
    print(f"【动/静态活动判定结果】")
    print(f"  - 静态活动代码 (Static Codes) : {static_codes}")
    print(f"  - 动态活动代码 (Dynamic Codes): {dynamic_codes}")
    print("="*50 + "\n")

    dom_freq_by_act, f_min = analyze_dominant_frequency(df_train_raw, dynamic_codes)
    L_min, L_max = compute_window_length_bounds(df_train_raw, f_min)

    print(f"\n" + "="*50)
    print(f"【动态活动主频分析结果】")
    for act, freq in dom_freq_by_act.items():
        print(f"  - 活动代码 {act:<2d} 的主频: {freq:.4f} Hz")
    print(f"  - 识别出的“最慢动态活动”主频 f_min = {f_min:.4f} Hz")
    print(f"  - 自动计算的窗口长度边界 (采样点数):")
    print(f"    * 满足最小循环周期(3周期)的最小窗口长度 L_min = {L_min} 帧 (约 {L_min/SAMPLING_RATE:.2f} 秒)")
    print(f"    * 基于样本长度约束的最大窗口长度 L_max = {L_max} 帧 (约 {L_max/SAMPLING_RATE:.2f} 秒)")
    print("="*50 + "\n")

    L_star, train_step_star, best_overlap = select_window_length_and_step(df_train_raw, L_min, L_max)

    print(f"\n" + "="*50)
    print(f"【最终选定的窗口与步长参数】")
    print(f"  - 最终窗口长度 L_star      : {L_star} 帧 (约 {L_star/SAMPLING_RATE:.2f} 秒)")
    print(f"  - 训练集滑动步长 step_star: {train_step_star} 帧")
    print(f"  - 选中的重叠比例 (Overlap) : {best_overlap * 100:.1f}%")
    print(f"  → 结论：窗口长度为 {L_star} 个采样点(约 {L_star/SAMPLING_RATE:.2f} 秒)，训练集步长为 {train_step_star}。")
    print("="*50 + "\n")

    channel_means, channel_stds = compute_channel_standardization(df_train_raw)
    df_train_std = apply_channel_standardization(df_train_raw, channel_means, channel_stds)
    df_test_std = apply_channel_standardization(df_test_raw, channel_means, channel_stds)

    static_means, static_stds = compute_static_covariate_standardization(df, train_ids)
    static_table_train = build_static_covariate_table(df, train_ids, static_means, static_stds)
    static_table_test = build_static_covariate_table(df, test_ids, static_means, static_stds)
    static_table_all = {**static_table_train, **static_table_test}

    train_windows = make_windows(df_train_std, L_star, train_step_star)
    test_windows = make_windows(df_test_std, L_star, L_star)

    print(f"\n[窗口统计] 训练集 = {len(train_windows)} 窗口, 测试集 = {len(test_windows)} 窗口")

    # 并行预计算
    precomputed_train = precompute_all_projections(train_windows, L_star, M_CANDIDATES, n_jobs=-1)
    precomputed_test = precompute_all_projections(test_windows, L_star, M_CANDIDATES, n_jobs=-1)

    # 路径A: 全局M
    M_star, global_M_results = select_global_M_fast(
        train_windows, precomputed_train, static_table_all, train_ids
    )

    X_train_A, y_train_A, id_train_A, gb_A, gn_A = build_design_matrix_global_M_fast(
        train_windows, precomputed_train, M_star, static_table_all
    )
    X_test_A, y_test_A, id_test_A, _, _ = build_design_matrix_global_M_fast(
        test_windows, precomputed_test, M_star, static_table_all
    )

    np.savez(
        os.path.join(OUTPUT_DIR, "design_matrix_global_M.npz"),
        X_train=X_train_A, y_train=y_train_A, id_train=id_train_A,
        X_test=X_test_A, y_test=y_test_A, id_test=id_test_A,
        group_boundaries=gb_A, group_names=gn_A,
        M_star=M_star, L_star=L_star,
    )

    # 路径B: 逐通道M_k
    M_k_dict, per_channel_results = select_per_channel_M_fast(
        train_windows, precomputed_train, static_table_all, train_ids
    )

    X_train_B, y_train_B, id_train_B, gb_B, gn_B = build_design_matrix_per_channel_Mk_fast(
        train_windows, precomputed_train, M_k_dict, static_table_all
    )
    X_test_B, y_test_B, id_test_B, _, _ = build_design_matrix_per_channel_Mk_fast(
        test_windows, precomputed_test, M_k_dict, static_table_all
    )

    np.savez(
        os.path.join(OUTPUT_DIR, "design_matrix_per_channel_Mk.npz"),
        X_train=X_train_B, y_train=y_train_B, id_train=id_train_B,
        X_test=X_test_B, y_test=y_test_B, id_test=id_test_B,
        group_boundaries=gb_B, group_names=gn_B,
        M_k_dict=json.dumps(M_k_dict), L_star=L_star,
    )

    print(f"\n{'='*70}\n[完成] 预处理完成，生成两个高效 design matrix！")
    print(f"路径 A 特征维度: {X_train_A.shape[1]}")
    print(f"路径 B 特征维度: {X_train_B.shape[1]}\n{'='*70}")


if __name__ == "__main__":
    main()