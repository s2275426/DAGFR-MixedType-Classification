"""
select_M_by_downstream_accuracy.py
========================================================================
用下游分类准确率(而非GCV重构误差)选择全局样条基维度M*

逻辑: GCV最小化的是"样条函数逼近原始曲线的精度", 这和DAGFR真正关心的
"样条系数能否被判别分类"是两个不同目标, 后者通常需要更强的平滑(更小的M)。
本脚本在17个训练集受试者内部做多次subject级别的子训练/子验证划分, 用带
L2正则的多元逻辑回归作为DAGFR的计算代理(同样是在B样条系数空间上做线性
判别), 对每个候选M测量平均验证准确率, 用one-standard-error规则选择"验证
准确率在1个标准误内, 且M最小"的候选, 全程不触碰真正的7个测试集受试者。
========================================================================
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from collections import Counter

# ---- 候选M网格: 覆盖低到高, 密度足够看出准确率随M变化的完整趋势 ----
M_CANDIDATES = [8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 72, 80, 84]
N_REPEATS = 15          # 每个M重复多少次随机subject划分, 降低偶然性
N_VAL_SUBJECTS = 5      # 17个训练集受试者中, 每次划出几人做子验证集
PROXY_C = 1.0           # 逻辑回归正则强度(代理分类器, 不是最终DAGFR)
SEED = 42


def build_coeffs_for_M(train_windows, L, M, degree=3):
    """对给定M重新做B样条展开, 返回(N, 12*M)的系数矩阵, 及对应id/act标签。"""
    from motionsense_preprocess import build_bspline_basis  # 复用已有函数
    B = build_bspline_basis(L, M, degree)
    B_pinv = np.linalg.pinv(B)

    X, y, ids = [], [], []
    for w in train_windows:
        coeffs = B_pinv @ w["data"]           # (M, 12)
        X.append(coeffs.T.reshape(-1))         # 展平为12*M维
        y.append(w["act"])
        ids.append(w["id"])
    return np.array(X), np.array(y), np.array(ids)


def evaluate_M_downstream(train_windows, L, M, train_subject_ids,
                           n_repeats=N_REPEATS, n_val_subjects=N_VAL_SUBJECTS,
                           seed=SEED):
    """对单个候选M, 做n_repeats次subject级别子划分, 返回验证准确率列表。"""
    X_all, y_all, id_all = build_coeffs_for_M(train_windows, L, M)
    rng = np.random.RandomState(seed)
    accs = []

    for rep in range(n_repeats):
        val_subj = rng.choice(train_subject_ids, size=n_val_subjects, replace=False)
        val_mask = np.isin(id_all, val_subj)
        tr_mask = ~val_mask

        X_tr, y_tr = X_all[tr_mask], y_all[tr_mask]
        X_val, y_val = X_all[val_mask], y_all[val_mask]

        # 特征标准化(在子训练集上fit, 应用到子验证集, 避免二次信息泄漏)
        scaler = StandardScaler()
        X_tr_std = scaler.fit_transform(X_tr)
        X_val_std = scaler.transform(X_val)

        clf = LogisticRegression(
            penalty="l2", C=PROXY_C, max_iter=2000, solver="lbfgs"
        )
        clf.fit(X_tr_std, y_tr)
        acc = clf.score(X_val_std, y_val)
        accs.append(acc)

    return accs


def select_M_one_se_rule(results, M_candidates):
    """
    one-standard-error规则:
    在"平均验证准确率最大值 - 1个标准误"以上的候选中, 选择M最小的那个,
    等价于"愿意为了下游准确率放弃部分重构精度, 但不做无意义的过大M"。
    """
    means = np.array([np.mean(results[M]) for M in M_candidates])
    ses = np.array([np.std(results[M], ddof=1) / np.sqrt(len(results[M]))
                     for M in M_candidates])

    best_idx = int(np.argmax(means))
    threshold = means[best_idx] - ses[best_idx]

    print(f"\n候选M的平均验证准确率(±标准误):")
    for M, m, s in zip(M_candidates, means, ses):
        marker = ""
        if m == means[best_idx]:
            marker = " ← 准确率最高"
        print(f"  M={M:<4d} 验证准确率 = {m:.4f} ± {s:.4f}{marker}")

    print(f"\n  最高准确率candidate: M={M_candidates[best_idx]}, "
          f"准确率={means[best_idx]:.4f}")
    print(f"  one-SE阈值线 = {threshold:.4f} "
          f"(最高准确率 - 1个标准误)")

    eligible = [M for M, m in zip(M_candidates, means) if m >= threshold]
    M_star = min(eligible)
    print(f"  → 满足'准确率不低于阈值线'的候选M = {eligible}")
    print(f"  → 按one-SE规则(优先选最小M以降低共线性和计算成本), "
          f"最终选定 M* = {M_star}")
    return M_star, means, ses


if __name__ == "__main__":
    # ---- 前置: 需要复用预处理脚本里已经生成的train_windows, L_star, train_ids ----
    # 假设你已经把 motionsense_preprocess.py 的主流程跑到"窗口切分"这一步,
    # 得到 train_windows(标准化后的训练集窗口列表)和 L_star(=160)和train_ids(17人)
    #
    # 下面这部分伪代码演示如何接入, 实际使用时直接在预处理脚本内部,
    # 在"窗口切分"完成之后、"决策点④"之前插入这段逻辑即可

    from motionsense_preprocess import (
        load_combined_data, select_subject_split, identify_static_dynamic_activities,
        analyze_dominant_frequency, compute_window_length_bounds,
        select_window_length_and_step, compute_channel_standardization,
        apply_channel_standardization, make_windows, RAW_CSV_PATH
    )

    df = load_combined_data(RAW_CSV_PATH)
    train_ids, test_ids = select_subject_split(df, 7)
    df_train_raw = df[df["id"].isin(train_ids)].reset_index(drop=True)

    static_codes, dynamic_codes = identify_static_dynamic_activities(df_train_raw)
    freq_results, f_min = analyze_dominant_frequency(df_train_raw, dynamic_codes)
    L_min, L_max = compute_window_length_bounds(df_train_raw, f_min)
    L_star, train_step_star, _ = select_window_length_and_step(df_train_raw, L_min, L_max)

    channel_means, channel_stds = compute_channel_standardization(df_train_raw)
    df_train_std = apply_channel_standardization(df_train_raw, channel_means, channel_stds)
    train_windows = make_windows(df_train_std, L_star, train_step_star)

    print("=" * 70)
    print(f"用下游分类准确率(代理分类器)为{len(M_CANDIDATES)}个候选M做评估")
    print(f"(每个M重复{N_REPEATS}次随机subject划分, 每次划出{N_VAL_SUBJECTS}人做验证)")
    print("=" * 70)

    results = {}
    for M in M_CANDIDATES:
        accs = evaluate_M_downstream(train_windows, L_star, M, train_ids)
        results[M] = accs
        print(f"  M={M:<4d} 完成, 平均验证准确率 = {np.mean(accs):.4f}")

    M_star, means, ses = select_M_one_se_rule(results, M_CANDIDATES)

    print(f"\n{'='*70}")
    print(f"最终决策: 基于下游分类准确率选定 M* = {M_star}")
    print(f"(该M将替代原GCV曲线选出的M=84, 用于重新构建design matrix)")
    print(f"{'='*70}")
