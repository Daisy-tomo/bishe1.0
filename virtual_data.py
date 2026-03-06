"""
虚拟试验数据生成模块
Virtual Experimental Data Generator

基于Mathcad计算流程：
  1. 给定增压比 π_c 和涡轮入口温度 T3 的范围
  2. 计算对应的比推力和耗油率
  3. 绘制性能图，选取"比推力大且耗油率低"的工作点
  4. 在选定工作点处添加高斯噪声，生成虚拟观测数据

观测数据模型（对应"二、观测数据模型"）：
    y_obs = y_model + ε，  ε ~ N(0, σ²)
即：
    F_obs   = F_model   + ε_F,    ε_F   ~ N(0, σ_F²)
    SFC_obs = SFC_model + ε_SFC,  ε_SFC ~ N(0, σ_SFC²)
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from turbojet_engine import TurbojetEngine

# ── 中文字体配置 ──────────────────────────────────────────────────────────
matplotlib.rcParams['font.family']         = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus']  = False


# ============================================================
# 1. 性能参数图（参数选优）
# ============================================================

def plot_performance_maps(engine: TurbojetEngine,
                          eta_c: float = 0.85,
                          eta_t: float = 0.88,
                          save_path: str = 'performance_maps.png') -> None:
    """
    绘制比推力与比油耗随增压比 π_c、涡轮入口温度 T3 变化的性能图。

    通过对比曲线，选择"比推力较大时耗油率较小"的 (π_c, T3) 组合，
    作为原型机工作参数及后续虚拟观测点的来源。

    Parameters
    ----------
    engine    : TurbojetEngine 实例
    eta_c     : 压气机效率（参考"真实值"）
    eta_t     : 涡轮效率（参考"真实值"）
    save_path : 保存路径
    """
    print("─" * 55)
    print("【步骤 1】绘制发动机性能参数图（选优工作参数）")
    print("─" * 55)

    # 扫描范围
    pi_range = np.linspace(4.0, 14.0, 200)
    T3_values = [1100, 1200, 1300, 1400, 1500]   # K
    colors    = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    lw = 2.2

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('涡喷发动机性能参数图\n'
                 f'(η_c={eta_c:.2f}, η_t={eta_t:.2f})',
                 fontsize=14, fontweight='bold')

    # ── 比推力 ──────────────────────────────────────────────
    ax1 = axes[0]
    for T3, col in zip(T3_values, colors):
        F_list, pi_valid = [], []
        for pi_c in pi_range:
            F, _, ok = engine.compute_performance(pi_c, T3, eta_c, eta_t)
            if ok:
                F_list.append(F)
                pi_valid.append(pi_c)
        if F_list:
            ax1.plot(pi_valid, F_list, color=col, lw=lw,
                     label=f'T₃ = {T3} K')

    ax1.set_xlabel('压气机增压比 π_c', fontsize=12)
    ax1.set_ylabel('比推力 F_sp  [N/(kg/s)]', fontsize=12)
    ax1.set_title('比推力 vs. 增压比', fontsize=13)
    ax1.legend(fontsize=10, framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([4, 14])

    # ── 比油耗 ──────────────────────────────────────────────
    ax2 = axes[1]
    for T3, col in zip(T3_values, colors):
        SFC_list, pi_valid = [], []
        for pi_c in pi_range:
            _, SFC, ok = engine.compute_performance(pi_c, T3, eta_c, eta_t)
            if ok:
                SFC_list.append(SFC * 1e6)   # 转为 mg/(N·s)
                pi_valid.append(pi_c)
        if SFC_list:
            ax2.plot(pi_valid, SFC_list, color=col, lw=lw,
                     label=f'T₃ = {T3} K')

    ax2.set_xlabel('压气机增压比 π_c', fontsize=12)
    ax2.set_ylabel('比油耗 SFC  [mg/(N·s)]', fontsize=12)
    ax2.set_title('比油耗 vs. 增压比', fontsize=13)
    ax2.legend(fontsize=10, framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([4, 14])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  ✓ 性能参数图已保存：{save_path}")
    print("  结论：T3 越高、π_c 适中时比推力最大；π_c 存在最优耗油率点。")


# ============================================================
# 2. 最优工作点对比图（π_c–T3 组合优选）
# ============================================================

def plot_optimal_region(engine: TurbojetEngine,
                        eta_c: float = 0.85,
                        eta_t: float = 0.88,
                        save_path: str = 'optimal_region.png') -> None:
    """
    绘制 (π_c, T3) 参数空间中的比推力与比油耗云图，
    直观显示"高推力低油耗"的最优区域。
    """
    print("\n  绘制最优工作区域云图…")

    pi_arr = np.linspace(4.0, 14.0, 80)
    T3_arr = np.linspace(1000.0, 1600.0, 60)
    PI, T3M = np.meshgrid(pi_arr, T3_arr)

    F_map   = np.full(PI.shape, np.nan)
    SFC_map = np.full(PI.shape, np.nan)

    for i in range(PI.shape[0]):
        for j in range(PI.shape[1]):
            F, SFC, ok = engine.compute_performance(
                PI[i, j], T3M[i, j], eta_c, eta_t)
            if ok:
                F_map[i, j]   = F
                SFC_map[i, j] = SFC * 1e6

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('(π_c, T₃) 参数空间性能云图', fontsize=13, fontweight='bold')

    for ax, data, title, cmap, label in [
        (axes[0], F_map,   '比推力 F_sp [N/(kg/s)]',  'viridis', 'F_sp'),
        (axes[1], SFC_map, '比油耗 SFC [mg/(N·s)]',   'RdYlGn_r', 'SFC')
    ]:
        cf = ax.contourf(PI, T3M, data, levels=20, cmap=cmap)
        plt.colorbar(cf, ax=ax, label=label)
        ax.contour(PI, T3M, data, levels=10, colors='black',
                   alpha=0.35, linewidths=0.7)
        ax.set_xlabel('增压比 π_c', fontsize=11)
        ax.set_ylabel('涡轮入口温度 T₃ [K]', fontsize=11)
        ax.set_title(title, fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  ✓ 最优工作区域图已保存：{save_path}")


# ============================================================
# 3. 虚拟观测数据生成
# ============================================================

# 根据性能图分析，选取工作点：
#   - 高比推力区：T3 ∈ [1200, 1450] K
#   - 较优耗油率：π_c ∈ [6, 11]
OPERATING_POINTS = [
    # (π_c,  T3 [K])   ← 从性能图中选取的典型工作点
    ( 6.0,  1200.0),
    ( 8.0,  1200.0),
    (10.0,  1200.0),
    ( 6.0,  1300.0),
    ( 8.0,  1300.0),
    (10.0,  1300.0),
    ( 7.0,  1350.0),
    ( 9.0,  1350.0),
    (11.0,  1350.0),
    ( 6.0,  1400.0),
    ( 8.0,  1400.0),
    (10.0,  1400.0),
    ( 7.0,  1450.0),
    ( 9.0,  1450.0),
    (11.0,  1450.0),
]


def generate_virtual_data(engine: TurbojetEngine,
                          eta_c_true: float = 0.85,
                          eta_t_true: float = 0.88,
                          noise_pct: float  = 0.01,
                          seed: int         = 42,
                          operating_points  = None) -> dict:
    """
    生成虚拟试验数据。

    流程：
      1. 用已知真实效率 (η_c_true, η_t_true) 计算各工作点理论性能；
      2. 添加零均值高斯噪声，模拟实际测量误差；
      3. 返回带噪声的观测数据集。

    Parameters
    ----------
    engine          : TurbojetEngine 实例
    eta_c_true      : 压气机效率真实值
    eta_t_true      : 涡轮效率真实值
    noise_pct       : 噪声水平（标准差占均值的百分比），默认 1%
    seed            : 随机种子
    operating_points: 工作点列表 [(π_c, T3), ...]，None 时使用内置列表

    Returns
    -------
    dict 包含：
        pi_c, T3       : 工作点输入
        F_obs, SFC_obs : 含噪声观测值
        F_true, SFC_true : 无噪声理论值
        sigma_F, sigma_SFC : 噪声标准差
        eta_c_true, eta_t_true, N
    """
    print("\n" + "─" * 55)
    print("【步骤 2】生成虚拟试验数据")
    print("─" * 55)

    if operating_points is None:
        operating_points = OPERATING_POINTS

    pi_c_arr = np.array([p[0] for p in operating_points])
    T3_arr   = np.array([p[1] for p in operating_points])
    N        = len(operating_points)

    # ── 计算无噪声理论性能 ────────────────────────────────────
    F_true   = np.zeros(N)
    SFC_true = np.zeros(N)
    for i, (pi_c, T3) in enumerate(operating_points):
        F, SFC, ok = engine.compute_performance(
            pi_c, T3, eta_c_true, eta_t_true)
        if not ok:
            raise ValueError(
                f"工作点 (π_c={pi_c}, T3={T3}) 在给定效率下计算失败！"
                f"请检查参数范围。")
        F_true[i]   = F
        SFC_true[i] = SFC

    # ── 添加高斯测量噪声 ─────────────────────────────────────
    rng = np.random.default_rng(seed)
    sigma_F   = noise_pct * np.mean(F_true)
    sigma_SFC = noise_pct * np.mean(SFC_true)

    F_obs   = F_true   + rng.normal(0.0, sigma_F,   N)
    SFC_obs = SFC_true + rng.normal(0.0, sigma_SFC, N)

    # ── 输出摘要 ─────────────────────────────────────────────
    print(f"  真实效率参数：η_c = {eta_c_true:.4f}，η_t = {eta_t_true:.4f}")
    print(f"  工作点数量：N = {N}")
    print(f"  测量噪声水平：{noise_pct*100:.1f}%")
    print(f"  σ_F   = {sigma_F:.4f} N/(kg/s)")
    print(f"  σ_SFC = {sigma_SFC*1e6:.5f} mg/(N·s)")
    print()
    print(f"  {'序号':>4} {'π_c':>6} {'T3[K]':>7} "
          f"{'F_obs[N/(kg/s)]':>17} {'SFC_obs[mg/(N·s)]':>19}")
    print("  " + "─" * 57)
    for i in range(N):
        print(f"  {i+1:>4} {pi_c_arr[i]:>6.1f} {T3_arr[i]:>7.0f} "
              f"{F_obs[i]:>17.3f} {SFC_obs[i]*1e6:>19.5f}")

    return dict(
        pi_c=pi_c_arr,
        T3=T3_arr,
        F_obs=F_obs,
        SFC_obs=SFC_obs,
        F_true=F_true,
        SFC_true=SFC_true,
        sigma_F=sigma_F,
        sigma_SFC=sigma_SFC,
        eta_c_true=eta_c_true,
        eta_t_true=eta_t_true,
        N=N,
    )


# ============================================================
# 4. 观测数据可视化
# ============================================================

def plot_observations(data: dict,
                      save_path: str = 'observations.png') -> None:
    """绘制虚拟观测数据，展示各工作点的性能分布"""
    print(f"\n  绘制虚拟观测数据图…")

    pi_c  = data['pi_c']
    T3    = data['T3']
    F_obs = data['F_obs']
    SFC_obs = data['SFC_obs'] * 1e6  # mg/(N·s)
    F_true  = data['F_true']
    SFC_true = data['SFC_true'] * 1e6
    sigma_F   = data['sigma_F']
    sigma_SFC = data['sigma_SFC'] * 1e6

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('虚拟试验观测数据（含高斯测量噪声）', fontsize=13, fontweight='bold')

    T3_unique = np.unique(T3)
    cmap_pts  = plt.cm.get_cmap('viridis', len(T3_unique))

    for ax, y_obs, y_true, sig, ylabel, title in [
        (axes[0], F_obs,   F_true,   sigma_F,   'F_sp [N/(kg/s)]',  '比推力观测数据'),
        (axes[1], SFC_obs, SFC_true, sigma_SFC, 'SFC [mg/(N·s)]',   '比油耗观测数据')
    ]:
        for k, T3_val in enumerate(T3_unique):
            mask = (T3 == T3_val)
            # 理论值（线）
            idx = np.argsort(pi_c[mask])
            ax.plot(pi_c[mask][idx], y_true[mask][idx],
                    color=cmap_pts(k), lw=1.8, alpha=0.7,
                    label=f'T₃={T3_val:.0f}K 理论')
            # 观测点（带误差棒）
            ax.errorbar(pi_c[mask], y_obs[mask],
                        yerr=2*sig, fmt='o', color=cmap_pts(k),
                        capsize=4, markersize=6, alpha=0.9,
                        label=f'T₃={T3_val:.0f}K 观测')

        ax.set_xlabel('增压比 π_c', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.grid(True, alpha=0.3)

    # 单独图例
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[:len(T3_unique)*2],
               labels[:len(T3_unique)*2],
               loc='lower center', ncol=len(T3_unique),
               fontsize=8, bbox_to_anchor=(0.5, -0.05))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  ✓ 观测数据图已保存：{save_path}")


# ============================================================
# 单独运行测试
# ============================================================

if __name__ == '__main__':
    eng  = TurbojetEngine()
    plot_performance_maps(eng)
    plot_optimal_region(eng)
    data = generate_virtual_data(eng)
    plot_observations(data)
