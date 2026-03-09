"""
贝叶斯反演可行性验证 — 代理前向模型 + 网格法主演示脚本
Bayesian Inversion Feasibility Demo — Surrogate Model + Grid Posterior

本脚本是 Mathcad 模板的完整 Python/NumPy 对应实现。
按 Mathcad 模板结构分为五步，输出图像与 Mathcad 绘图结果一一对应。

运行方式：
    python run_grid_demo.py                  # 默认参数
    python run_grid_demo.py --outdir my_out  # 指定输出目录
    python run_grid_demo.py --pi-true 32 --Tg-true 1550 --m-true 7
    python run_grid_demo.py --N-pi 30 --N-T 30 --N-m 25  # 更精细网格

输出图像（对应 Mathcad 第八节图形输出）：
    surrogate_model_check.png   — 代理模型参数敏感性扫描（6子图）
    posterior_marginals_1d.png  — 一维边缘后验 p(π_k|y), p(T_g|y), p(m|y)
    posterior_contours_2d.png   — 二维联合后验等高线图（3组）
    inversion_summary.png       — 反演结果汇总（参数对比 + 数值表 + 后验切片）
"""

import os
import argparse
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from turbofan_surrogate import TurbofanSurrogate
from grid_bayes import GridBayesianInversion

matplotlib.rcParams['font.family']        = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


# ==============================================================
# 命令行参数
# ==============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description='涡扇发动机循环参数贝叶斯反演可行性验证（代理模型网格法）')
    # 真值参数
    p.add_argument('--pi-true',  type=float, default=30.0,
                   help='π_k 真值（默认 30.0）')
    p.add_argument('--Tg-true',  type=float, default=1500.0,
                   help='T_g 真值 [K]（默认 1500.0）')
    p.add_argument('--m-true',   type=float, default=8.0,
                   help='m 真值（默认 8.0）')
    # 虚拟观测扰动
    p.add_argument('--dR',       type=float, default=5.0,
                   help='比推力固定扰动 ΔR [N·s/kg]（默认 5.0）')
    p.add_argument('--dC',       type=float, default=5e-4,
                   help='比油耗固定扰动 ΔC [kg/(N·h)]（默认 5e-4）')
    # 似然不确定度
    p.add_argument('--sigma-R',  type=float, default=10.0,
                   help='σ_R [N·s/kg]（默认 10.0）')
    p.add_argument('--sigma-C',  type=float, default=1e-3,
                   help='σ_C [kg/(N·h)]（默认 1e-3）')
    # 网格分辨率
    p.add_argument('--N-pi',     type=int, default=25,
                   help='π_k 网格点数（默认 25）')
    p.add_argument('--N-T',      type=int, default=25,
                   help='T_g 网格点数（默认 25）')
    p.add_argument('--N-m',      type=int, default=20,
                   help='m 网格点数（默认 20）')
    # 输出
    p.add_argument('--outdir',   type=str, default='results_grid',
                   help='输出目录（默认 results_grid）')
    return p.parse_args()


def _path(outdir, fname):
    os.makedirs(outdir, exist_ok=True)
    return os.path.join(outdir, fname)


# ==============================================================
# 绘图函数
# ==============================================================

def plot_surrogate_check(model, outdir):
    """
    绘制代理模型参数敏感性扫描图（6子图）

    对应 Mathcad 第一节代理前向模型的参数趋势验证。
    行：R_ud、C_ud；列：π_k 扫描、T_g 扫描、m 扫描
    """
    print("  [图1] 代理模型参数敏感性扫描…")

    pi_ref, T_ref, m_ref = 30.0, 1500.0, 8.0
    pi_arr = np.linspace(15, 45, 200)
    T_arr  = np.linspace(1200, 1800, 200)
    m_arr  = np.linspace(4, 12, 200)

    T_vals = [1200, 1350, 1500, 1650, 1800]
    m_vals = [4, 6, 8, 10, 12]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(
        '代理前向模型参数敏感性分析\n'
        r'$R_{ud}=500\cdot(T_g/1000)^{0.8}\cdot\pi_k^{0.25}/(1+0.25m)$'
        r'   $C_{ud}=0.06\cdot(T_g/1200)^{1.1}/[\pi_k^{0.15}(1+0.05m)]$',
        fontsize=10, fontweight='bold')

    # ── 行 0：R_ud ───────────────────────────────────────────────
    ax = axes[0, 0]
    for T_v, col in zip(T_vals, colors):
        ax.plot(pi_arr, model.compute_R_ud(pi_arr, T_v, m_ref),
                color=col, lw=2, label=f'T_g={T_v} K')
    ax.axvline(pi_ref, color='gray', ls='--', lw=1, alpha=0.7)
    ax.set_xlabel('压气机总压比 π_k', fontsize=9)
    ax.set_ylabel('比推力 R_ud  [N·s/kg]', fontsize=9)
    ax.set_title('R_ud vs. π_k  (m=8)', fontsize=10)
    ax.legend(fontsize=7.5); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    for m_v, col in zip(m_vals, colors):
        ax.plot(T_arr, model.compute_R_ud(pi_ref, T_arr, m_v),
                color=col, lw=2, label=f'm={m_v}')
    ax.axvline(T_ref, color='gray', ls='--', lw=1, alpha=0.7)
    ax.set_xlabel('涡轮进口温度 T_g  [K]', fontsize=9)
    ax.set_ylabel('比推力 R_ud  [N·s/kg]', fontsize=9)
    ax.set_title('R_ud vs. T_g  (π_k=30)', fontsize=10)
    ax.legend(fontsize=7.5); ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    for T_v, col in zip(T_vals, colors):
        ax.plot(m_arr, model.compute_R_ud(pi_ref, T_v, m_arr),
                color=col, lw=2, label=f'T_g={T_v} K')
    ax.axvline(m_ref, color='gray', ls='--', lw=1, alpha=0.7)
    ax.set_xlabel('涵道比 m', fontsize=9)
    ax.set_ylabel('比推力 R_ud  [N·s/kg]', fontsize=9)
    ax.set_title('R_ud vs. m  (π_k=30)', fontsize=10)
    ax.legend(fontsize=7.5); ax.grid(True, alpha=0.3)

    # ── 行 1：C_ud ───────────────────────────────────────────────
    ax = axes[1, 0]
    for T_v, col in zip(T_vals, colors):
        ax.plot(pi_arr, model.compute_C_ud(pi_arr, T_v, m_ref) * 1e3,
                color=col, lw=2, label=f'T_g={T_v} K')
    ax.axvline(pi_ref, color='gray', ls='--', lw=1, alpha=0.7)
    ax.set_xlabel('压气机总压比 π_k', fontsize=9)
    ax.set_ylabel('C_ud  [×10⁻³ kg/(N·h)]', fontsize=9)
    ax.set_title('C_ud vs. π_k  (m=8)', fontsize=10)
    ax.legend(fontsize=7.5); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    for m_v, col in zip(m_vals, colors):
        ax.plot(T_arr, model.compute_C_ud(pi_ref, T_arr, m_v) * 1e3,
                color=col, lw=2, label=f'm={m_v}')
    ax.axvline(T_ref, color='gray', ls='--', lw=1, alpha=0.7)
    ax.set_xlabel('涡轮进口温度 T_g  [K]', fontsize=9)
    ax.set_ylabel('C_ud  [×10⁻³ kg/(N·h)]', fontsize=9)
    ax.set_title('C_ud vs. T_g  (π_k=30)', fontsize=10)
    ax.legend(fontsize=7.5); ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    for T_v, col in zip(T_vals, colors):
        ax.plot(m_arr, model.compute_C_ud(pi_ref, T_v, m_arr) * 1e3,
                color=col, lw=2, label=f'T_g={T_v} K')
    ax.axvline(m_ref, color='gray', ls='--', lw=1, alpha=0.7)
    ax.set_xlabel('涵道比 m', fontsize=9)
    ax.set_ylabel('C_ud  [×10⁻³ kg/(N·h)]', fontsize=9)
    ax.set_title('C_ud vs. m  (π_k=30)', fontsize=10)
    ax.legend(fontsize=7.5); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = _path(outdir, 'surrogate_model_check.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"     ✓ {save_path}")


def plot_marginal_posteriors(inv, pi_true, T_true, m_true,
                              map_res, stats, outdir):
    """
    绘制三个参数的一维边缘后验分布

    对应 Mathcad 第八节图形输出 1：
        p(π_k | y_obs), p(T_g | y_obs), p(m | y_obs)
    每幅子图标注：真值（红竖线）、MAP（紫虚线）、后验均值（橙点线）、95% CI（填色带）
    """
    print("  [图2] 一维边缘后验分布…")

    p_pi, p_T, p_m = inv.marginal_1d()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        '贝叶斯后验边缘分布  p(θ | y_obs)\n'
        '（代理前向模型 · 均匀先验 · 网格离散化）',
        fontsize=12, fontweight='bold')

    configs = [
        (axes[0], inv.pi_grid, p_pi, inv.d_pi,
         pi_true, map_res['pi_k_MAP'], stats['mu_pi'], stats['std_pi'],
         'π_k  压气机总压比', 'steelblue'),
        (axes[1], inv.T_grid, p_T, inv.d_T,
         T_true,  map_res['T_g_MAP'],  stats['mu_T'],  stats['std_T'],
         'T_g  涡轮进口温度 [K]', 'darkorange'),
        (axes[2], inv.m_grid, p_m, inv.d_m,
         m_true,  map_res['m_MAP'],    stats['mu_m'],  stats['std_m'],
         'm  涵道比', 'seagreen'),
    ]

    for ax, grid, p, dg, tv, map_v, mu, std, xlabel, col in configs:
        # 后验曲线
        ax.fill_between(grid, p, alpha=0.28, color=col)
        ax.plot(grid, p, color=col, lw=2.5, label='后验 p(θ|y_obs)')

        # 95% 可信区间（CDF 反查）
        cdf = np.cumsum(p) * dg
        cdf = np.clip(cdf, 0, 1)
        ci_lo = float(grid[np.searchsorted(cdf, 0.025, side='left').clip(0, len(grid)-1)])
        ci_hi = float(grid[np.searchsorted(cdf, 0.975, side='left').clip(0, len(grid)-1)])
        mask  = (grid >= ci_lo) & (grid <= ci_hi)
        ax.fill_between(grid[mask], p[mask], alpha=0.22, color=col,
                        label=f'95% CI [{ci_lo:.2f}, {ci_hi:.2f}]')

        # 标注线
        ax.axvline(tv,    color='red',    lw=2.5, ls='-',
                   label=f'真值  = {tv:.2f}')
        ax.axvline(map_v, color='purple', lw=2.0, ls='--',
                   label=f'MAP   = {map_v:.3f}')
        ax.axvline(mu,    color='orange', lw=1.5, ls=':',
                   label=f'后验均值 = {mu:.3f}')

        # 误差注释
        err_pct = abs(map_v - tv) / abs(tv) * 100
        ax.text(0.04, 0.94,
                f'MAP误差：{err_pct:.2f}%\n后验标准差：{std:.3f}',
                transform=ax.transAxes, fontsize=9, va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85))

        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel('概率密度', fontsize=10)
        ax.set_title(f'p({xlabel.split()[0]} | y_obs)', fontsize=11)
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = _path(outdir, 'posterior_marginals_1d.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"     ✓ {save_path}")


def plot_joint_posteriors(inv, pi_true, T_true, m_true, map_res, outdir):
    """
    绘制三组二维联合边缘后验等高线图

    对应 Mathcad 第八节图形输出 2：
        p(π_k, T_g | y_obs), p(π_k, m | y_obs), p(T_g, m | y_obs)
    """
    print("  [图3] 二维联合后验等高线图…")

    p_piT, p_pim, p_Tm = inv.marginal_2d()

    fig, axes = plt.subplots(1, 3, figsize=(17, 6))
    fig.suptitle(
        '二维联合边缘后验  p(θ₁, θ₂ | y_obs)\n'
        '（★ 红星=真值，▲ 黄三角=边缘MAP，颜色越深概率越高）',
        fontsize=12, fontweight='bold')

    # 每组的配置：(x_grid, y_grid, Z矩阵, 真值x, 真值y, MAPx, MAPy, xlabel, ylabel, title)
    configs = [
        (inv.pi_grid, inv.T_grid, p_piT.T,
         pi_true, T_true, map_res['pi_k_MAP'], map_res['T_g_MAP'],
         'π_k  压气机总压比', 'T_g  涡轮进口温度 [K]', 'p(π_k, T_g | y_obs)'),
        (inv.pi_grid, inv.m_grid, p_pim.T,
         pi_true, m_true, map_res['pi_k_MAP'], map_res['m_MAP'],
         'π_k  压气机总压比', 'm  涵道比', 'p(π_k, m | y_obs)'),
        (inv.T_grid, inv.m_grid, p_Tm.T,
         T_true, m_true, map_res['T_g_MAP'], map_res['m_MAP'],
         'T_g  涡轮进口温度 [K]', 'm  涵道比', 'p(T_g, m | y_obs)'),
    ]

    for ax, (x_g, y_g, Z, x_tv, y_tv, x_mp, y_mp, xl, yl, title) in zip(axes, configs):
        XX, YY = np.meshgrid(x_g, y_g)
        cf = ax.contourf(XX, YY, Z, levels=25, cmap='Blues', alpha=0.92)
        plt.colorbar(cf, ax=ax, label='联合后验概率密度', shrink=0.82)
        ax.contour(XX, YY, Z, levels=8,
                   colors='navy', alpha=0.35, linewidths=0.7)

        ax.plot(x_tv, y_tv, 'r*', ms=16, zorder=10,
                label=f'真值 ({x_tv:.1f}, {y_tv:.1f})')
        ax.plot(x_mp, y_mp, 'y^', ms=11,
                markeredgecolor='darkorange', markeredgewidth=1.5,
                zorder=10,
                label=f'MAP ({x_mp:.2f}, {y_mp:.2f})')

        ax.set_xlabel(xl, fontsize=9)
        ax.set_ylabel(yl, fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8, loc='lower right')
        ax.grid(True, alpha=0.18)

    plt.tight_layout()
    save_path = _path(outdir, 'posterior_contours_2d.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"     ✓ {save_path}")


def plot_inversion_summary(inv, pi_true, T_true, m_true,
                            R_true, C_true, R_obs, C_obs,
                            map_res, err, stats, outdir):
    """
    绘制贝叶斯反演结果汇总图（4子图）

    对应 Mathcad 第八节图形输出 3（真值与MAP对比表）+
    后验对数切片（π_k–T_g 截面，固定 m≈MAP）
    """
    print("  [图4] 反演结果汇总图…")

    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 3, hspace=0.48, wspace=0.38,
                            figure=fig)
    fig.suptitle(
        '贝叶斯反演结果汇总\n'
        '（代理前向模型  ·  均匀先验  ·  网格离散后验  ·  MAP 提取）',
        fontsize=13, fontweight='bold')

    # ── 子图1：循环参数真值 vs. MAP 柱状图 ──────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    labels  = ['π_k', 'T_g / 50', 'm × 10']
    v_true  = [pi_true, T_true / 50.0, m_true * 10.0]
    v_map   = [map_res['pi_k_MAP'],
               map_res['T_g_MAP'] / 50.0,
               map_res['m_MAP']   * 10.0]
    x = np.arange(len(labels))
    w = 0.34
    b_true = ax1.bar(x - w/2, v_true, w,
                     label='真值 θ_true', color='steelblue', alpha=0.85,
                     edgecolor='white', linewidth=0.5)
    b_map  = ax1.bar(x + w/2, v_map,  w,
                     label='MAP θ_MAP',  color='darkorange', alpha=0.85,
                     edgecolor='white', linewidth=0.5)
    ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=12)
    ax1.set_ylabel('归一化数值（便于同图对比）', fontsize=10)
    ax1.set_title('循环参数真值 vs. MAP 估计', fontsize=11)
    ax1.legend(fontsize=10); ax1.grid(True, alpha=0.3, axis='y')
    for bt, bm, vt, vm in zip(b_true, b_map, v_true, v_map):
        ax1.text(bt.get_x() + bt.get_width() / 2, vt + 0.4,
                 f'{vt:.2f}', ha='center', fontsize=9, color='steelblue')
        ax1.text(bm.get_x() + bm.get_width() / 2, vm + 0.4,
                 f'{vm:.2f}', ha='center', fontsize=9, color='darkorange')

    # ── 子图2：数值对比文本表格 ──────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis('off')
    rows = [
        ['π_k',     f'{pi_true:.2f}',
         f"{map_res['pi_k_MAP']:.3f}",
         f"{err['err_pi_pct']:+.2f}%"],
        ['T_g [K]', f'{T_true:.1f}',
         f"{map_res['T_g_MAP']:.2f}",
         f"{err['err_T_pct']:+.2f}%"],
        ['m',       f'{m_true:.2f}',
         f"{map_res['m_MAP']:.3f}",
         f"{err['err_m_pct']:+.2f}%"],
        ['', '', '', ''],
        ['R_ud',    f'{R_true:.2f}',
         f'{R_obs:.2f}',
         f"{err['R_MAP']:.2f}"],
        ['C_ud×10³', f'{C_true*1e3:.4f}',
         f'{C_obs*1e3:.4f}',
         f"{err['C_MAP']*1e3:.4f}"],
    ]
    tab = ax2.table(
        cellText=rows,
        colLabels=['参数/量', '真值', 'MAP/观测', '误差%'],
        loc='center', cellLoc='center')
    tab.auto_set_font_size(False)
    tab.set_fontsize(9)
    tab.scale(1.05, 1.65)
    for (r, c), cell in tab.get_celld().items():
        if r == 0:
            cell.set_facecolor('#4472C4')
            cell.set_text_props(color='white', fontweight='bold')
        elif r % 2 == 0 and r > 0:
            cell.set_facecolor('#E8F0FE')
    ax2.set_title('数值对比表', fontsize=11, pad=14)

    # ── 子图3：后验 π_k–T_g 切片（固定 m≈MAP）────────────────────
    ax3 = fig.add_subplot(gs[1, :2])
    idx_m_s = int(np.argmin(np.abs(inv.m_grid - map_res['m_MAP'])))
    m_slice = inv.m_grid[idx_m_s]
    # 用对数后验做切片（视觉上等高线更清晰）
    post_slice = inv._post_3d[:, :, idx_m_s].T   # shape (NT, Nπ)
    log_slice  = np.log(post_slice + 1e-300)
    log_slice -= log_slice.max()

    XX, YY = np.meshgrid(inv.pi_grid, inv.T_grid)
    cf = ax3.contourf(XX, YY, log_slice, levels=25,
                      cmap='RdYlGn', alpha=0.90)
    plt.colorbar(cf, ax=ax3, label='ln p(π_k, T_g | y_obs, m≈MAP)')
    ax3.contour(XX, YY, log_slice, levels=10,
                colors='black', alpha=0.25, linewidths=0.6)

    ax3.plot(pi_true, T_true, 'r*', ms=17, zorder=10,
             label=f'真值 ({pi_true:.0f}, {T_true:.0f})')
    ax3.plot(map_res['pi_k_MAP'], map_res['T_g_MAP'], 'y^', ms=12,
             markeredgecolor='darkorange', markeredgewidth=1.5,
             zorder=10,
             label=f"MAP ({map_res['pi_k_MAP']:.2f}, "
                   f"{map_res['T_g_MAP']:.1f})")
    ax3.set_xlabel('π_k  压气机总压比', fontsize=10)
    ax3.set_ylabel('T_g  涡轮进口温度 [K]', fontsize=10)
    ax3.set_title(f'后验对数切片  p(π_k, T_g | y, m={m_slice:.2f})',
                  fontsize=11)
    ax3.legend(fontsize=9); ax3.grid(True, alpha=0.2)

    # ── 子图4：性能量三组对比柱状图 ─────────────────────────────
    ax4 = fig.add_subplot(gs[1, 2])
    cat = ['R_ud\n[N·s/kg]', 'C_ud×10³\n[kg/(N·h)]']
    vt  = [R_true,     C_true * 1e3]
    vo  = [R_obs,      C_obs  * 1e3]
    vm  = [err['R_MAP'], err['C_MAP'] * 1e3]
    x4  = np.arange(len(cat))
    w4  = 0.25
    ax4.bar(x4 - w4,   vt, w4, label='真值',     color='steelblue',  alpha=0.85)
    ax4.bar(x4,        vo, w4, label='观测值',   color='coral',      alpha=0.85)
    ax4.bar(x4 + w4,   vm, w4, label='MAP 预测', color='darkorange', alpha=0.85)
    ax4.set_xticks(x4); ax4.set_xticklabels(cat, fontsize=9)
    ax4.set_title('性能量真值 / 观测 / MAP 对比', fontsize=10)
    ax4.legend(fontsize=8.5); ax4.grid(True, alpha=0.3, axis='y')

    save_path = _path(outdir, 'inversion_summary.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"     ✓ {save_path}")


# ==============================================================
# 主流程
# ==============================================================

def main():
    args = parse_args()

    print("=" * 70)
    print("  贝叶斯反演可行性验证 — 代理前向模型 + 网格后验")
    print("  Bayesian Inversion Feasibility Demo (Surrogate + Grid)")
    print("=" * 70)
    print(f"  真值参数 θ_true：π_k = {args.pi_true},  "
          f"T_g = {args.Tg_true} K,  m = {args.m_true}")
    print(f"  固定扰动：ΔR = {args.dR} N·s/kg,  "
          f"ΔC = {args.dC:.2e} kg/(N·h)")
    print(f"  似然不确定度：σ_R = {args.sigma_R},  "
          f"σ_C = {args.sigma_C:.2e}")
    print(f"  网格分辨率：{args.N_pi} × {args.N_T} × {args.N_m}"
          f" = {args.N_pi * args.N_T * args.N_m:,} 个网格点")
    print(f"  输出目录：{args.outdir}/")
    print("=" * 70)

    # ──────────────────────────────────────────────────────────────────
    # 第一步：初始化代理模型 & 参数扫描验证
    # ──────────────────────────────────────────────────────────────────
    print("\n" + "─" * 55)
    print("【第一步】代理前向模型初始化与趋势验证")
    print("─" * 55)

    model = TurbofanSurrogate()
    R_verify, C_verify = model.verify_trends()
    plot_surrogate_check(model, args.outdir)

    # ──────────────────────────────────────────────────────────────────
    # 第二步：真值参数 & 虚拟观测数据生成（固定扰动，无随机数）
    # ──────────────────────────────────────────────────────────────────
    print("\n" + "─" * 55)
    print("【第二步】真值参数与虚拟观测数据生成")
    print("─" * 55)

    pi_true = args.pi_true
    T_true  = args.Tg_true
    m_true  = args.m_true

    R_true = float(model.compute_R_ud(pi_true, T_true, m_true))
    C_true = float(model.compute_C_ud(pi_true, T_true, m_true))
    R_obs  = R_true + args.dR
    C_obs  = C_true + args.dC

    print(f"  θ_true = (π_k={pi_true:.2f},  T_g={T_true:.1f} K,  m={m_true:.2f})")
    print()
    print(f"  真值性能量：")
    print(f"    R_true = {R_true:.6f}  N·s/kg")
    print(f"    C_true = {C_true:.8f} kg/(N·h)")
    print()
    print(f"  固定扰动（模拟测量噪声，不使用随机函数）：")
    print(f"    ΔR = {args.dR:+.4f}  ({args.dR/R_true*100:+.2f}% of R_true)")
    print(f"    ΔC = {args.dC:+.2e}  ({args.dC/C_true*100:+.2f}% of C_true)")
    print()
    print(f"  虚拟观测量：")
    print(f"    R_obs = {R_obs:.6f}  N·s/kg")
    print(f"    C_obs = {C_obs:.8f} kg/(N·h)")

    # ──────────────────────────────────────────────────────────────────
    # 第三步：先验范围定义 & 网格后验计算
    # ──────────────────────────────────────────────────────────────────
    print("\n" + "─" * 55)
    print("【第三步】先验参数范围 & 网格后验计算")
    print("─" * 55)

    inv = GridBayesianInversion(model)
    inv.N_pi    = args.N_pi
    inv.N_T     = args.N_T
    inv.N_m     = args.N_m
    inv.sigma_R = args.sigma_R
    inv.sigma_C = args.sigma_C

    print(f"  先验范围（均匀先验）：")
    print(f"    π_k ∈ [{inv.pi_k_min}, {inv.pi_k_max}]")
    print(f"    T_g ∈ [{inv.T_g_min}, {inv.T_g_max}] K")
    print(f"    m   ∈ [{inv.m_min}, {inv.m_max}]")
    print()
    print(f"  计算 {inv.N_pi}×{inv.N_T}×{inv.N_m} 网格后验…", end='', flush=True)

    post_3d = inv.compute_posterior(R_obs, C_obs)
    print(f" 完成。")
    print(f"  后验 3D 数组形状：{post_3d.shape}")
    print(f"  后验最大值：{post_3d.max():.6e}")

    # 验证归一化（Mathcad 验证清单第4条）
    norm_check = post_3d.sum() * inv.d_pi * inv.d_T * inv.d_m
    print(f"  归一化验证：Σ post · Δπ · ΔT · Δm = {norm_check:.6f}  "
          f"（应 ≈ 1.0 {'✓' if abs(norm_check - 1.0) < 0.05 else '⚠'}）")

    # ──────────────────────────────────────────────────────────────────
    # 第四步：MAP 估计 & 误差分析
    # ──────────────────────────────────────────────────────────────────
    print("\n" + "─" * 55)
    print("【第四步】MAP 估计与误差分析")
    print("─" * 55)

    map_res = inv.compute_map()
    stats   = inv.posterior_stats()
    err     = inv.compute_errors(pi_true, T_true, m_true)

    inv.print_summary(pi_true, T_true, m_true, R_true, C_true)

    # 验证：MAP 误差应小于网格步长（Mathcad 验证清单第5条）
    print("\n  网格步长验证：")
    for name, e_abs, step in [
        ('π_k', err['err_pi_abs'], inv.d_pi),
        ('T_g', err['err_T_abs'],  inv.d_T),
        ('m',   err['err_m_abs'],  inv.d_m),
    ]:
        within = '✓' if abs(e_abs) <= step * 1.5 else '⚠（建议增加网格密度）'
        print(f"    |err_{name}| = {abs(e_abs):.4f}  vs  Δ = {step:.4f}  {within}")

    # ──────────────────────────────────────────────────────────────────
    # 第五步：可视化
    # ──────────────────────────────────────────────────────────────────
    print("\n" + "─" * 55)
    print("【第五步】可视化生成")
    print("─" * 55)

    plot_marginal_posteriors(inv, pi_true, T_true, m_true,
                              map_res, stats, args.outdir)
    plot_joint_posteriors(inv, pi_true, T_true, m_true,
                           map_res, args.outdir)
    plot_inversion_summary(inv, pi_true, T_true, m_true,
                            R_true, C_true, R_obs, C_obs,
                            map_res, err, stats, args.outdir)

    # ──────────────────────────────────────────────────────────────────
    # 完成
    # ──────────────────────────────────────────────────────────────────
    print("\n  输出文件清单：")
    files = [
        'surrogate_model_check.png',
        'posterior_marginals_1d.png',
        'posterior_contours_2d.png',
        'inversion_summary.png',
    ]
    for fname in files:
        fpath = _path(args.outdir, fname)
        status = '✓' if os.path.exists(fpath) else '✗'
        print(f"    [{status}] {fpath}")

    print("\n" + "=" * 70)
    print("  可行性验证完成！")
    print()
    print("  结论：贝叶斯反演方法在代理模型上成立，MAP 估计与真值接近。")
    print()
    print("  后续扩展步骤：")
    print("  1. 将 turbofan_surrogate.py 的 R_ud / C_ud 替换为")
    print("     真实热力循环公式（如 GasTurb 插值或解析循环方程）；")
    print("  2. 或接入 mcmc_sampler.py 使用 MCMC 做更精细的后验采样；")
    print("  3. 网格法可扩展至 4+ 参数（η_c、η_t 加入反演）。")
    print("=" * 70)


# ── 快速测试（CI 用）──────────────────────────────────────────────────

def quick_test():
    """冒烟测试：验证代理模型、网格后验、MAP 计算流程可运行"""
    print("── 代理模型贝叶斯反演快速测试 ──")

    model = TurbofanSurrogate()
    R, C, ok = model.compute_performance(30.0, 1500.0, 8.0)
    assert ok and R > 0 and C > 0, "代理模型计算失败"
    print(f"  代理模型  OK  R_ud={R:.2f},  C_ud={C:.6f}")

    inv = GridBayesianInversion(model)
    inv.N_pi = 10; inv.N_T = 10; inv.N_m = 8
    post = inv.compute_posterior(R + 5.0, C + 5e-4)
    assert post.shape == (10, 10, 8), "后验形状不正确"
    assert np.isfinite(post).all(), "后验包含非有限值"
    print(f"  网格后验  OK  shape={post.shape},  max={post.max():.4e}")

    map_res = inv.compute_map()
    assert 'pi_k_MAP' in map_res, "MAP 提取失败"
    print(f"  MAP 提取  OK  π_k={map_res['pi_k_MAP']:.2f},  "
          f"T_g={map_res['T_g_MAP']:.1f},  m={map_res['m_MAP']:.2f}")

    p_pi, p_T, p_m = inv.marginal_1d()
    assert len(p_pi) == 10, "边缘后验维度错误"
    print(f"  边缘后验  OK  p_pi.sum·dπ={p_pi.sum()*inv.d_pi:.4f}")

    print("── 所有测试通过 ✓ ──\n")
    return True


# ==============================================================

if __name__ == '__main__':
    import sys
    if '--test' in sys.argv:
        quick_test()
    else:
        main()
