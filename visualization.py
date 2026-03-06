"""
结果分析与可视化模块
Results Analysis & Visualization

生成以下全套图像：
  1. performance_maps.png      — 比推力 / 比油耗性能图
  2. optimal_region.png        — (π_c, T3) 空间最优工作区域
  3. observations.png          — 虚拟观测数据
  4. likelihood_surface.png    — 对数似然曲面 & 等高线
  5. mcmc_diagnostics.png      — MCMC 链图 & 自相关函数
  6. posterior_marginals.png   — 后验边缘分布（直方图 + KDE）
  7. posterior_contour.png     — 二维联合后验等高线图
  8. model_fit.png             — 后验预测 vs. 观测数据对比
  9. sensitivity.png           — 噪声水平 / 数据量敏感性分析
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import gaussian_kde
from turbojet_engine import TurbojetEngine
from bayesian_model import compute_likelihood_surface, log_posterior

matplotlib.rcParams['font.family']        = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
           '#8c564b', '#e377c2', '#7f7f7f']


# ============================================================
# 1. 对数似然曲面图
# ============================================================

def plot_likelihood_surface(data: dict,
                             engine: TurbojetEngine,
                             save_path: str = 'likelihood_surface.png'):
    """绘制对数似然函数在 (η_c, η_t) 参数空间的等高线图"""
    print("\n  [可视化] 绘制似然函数曲面…")

    eta_c_range = np.linspace(0.70, 0.97, 70)
    eta_t_range = np.linspace(0.70, 0.97, 70)
    ETA_C, ETA_T, LOG_L = compute_likelihood_surface(
        data, engine, eta_c_range, eta_t_range)

    # 最大似然估计点
    valid = np.isfinite(LOG_L)
    max_idx = np.unravel_index(np.nanargmax(LOG_L), LOG_L.shape)
    eta_c_mle = ETA_C[max_idx]
    eta_t_mle = ETA_T[max_idx]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('对数似然函数  ln P(D | η_c, η_t)',
                 fontsize=14, fontweight='bold')

    # ── 填充等高线 ─────────────────────────────────────────
    for ax, use_log in [(axes[0], False), (axes[1], True)]:
        data_plot = LOG_L if not use_log else LOG_L - np.nanmax(LOG_L)
        title     = '对数似然 ln L' if not use_log else '归一化对数似然 (最大值=0)'

        levels = np.nanpercentile(data_plot[valid],
                                   np.linspace(5, 100, 25))
        cf = ax.contourf(ETA_C, ETA_T, data_plot,
                         levels=levels, cmap='RdYlGn', alpha=0.85)
        plt.colorbar(cf, ax=ax, label='ln L')
        ax.contour(ETA_C, ETA_T, data_plot,
                   levels=12, colors='black', alpha=0.35, linewidths=0.7)

        ax.plot(eta_c_mle, eta_t_mle, 'k^', ms=12, zorder=10,
                label=f'MLE ({eta_c_mle:.4f}, {eta_t_mle:.4f})')
        ax.plot(data['eta_c_true'], data['eta_t_true'],
                'r*', ms=16, zorder=11,
                label=f'真实值 ({data["eta_c_true"]:.4f}, '
                      f'{data["eta_t_true"]:.4f})')
        ax.set_xlabel('η_c  压气机等熵效率', fontsize=12)
        ax.set_ylabel('η_t  涡轮等熵效率',   fontsize=12)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=9, loc='lower right')
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  ✓ {save_path}")


# ============================================================
# 2. MCMC 诊断图
# ============================================================

def plot_mcmc_diagnostics(all_samples: np.ndarray,
                           info: dict,
                           data: dict,
                           save_path: str = 'mcmc_diagnostics.png'):
    """
    MCMC 链图、自相关函数 (ACF)、累积后验均值收敛图
    """
    print("\n  [可视化] 绘制 MCMC 诊断图…")

    n_burnin  = info['n_burnin']
    post_samp = all_samples[n_burnin:]
    param_names = ['η_c  压气机效率', 'η_t  涡轮效率']
    true_vals   = [data['eta_c_true'], data['eta_t_true']]
    cols        = ['steelblue', 'seagreen']

    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(3, 2, hspace=0.45, wspace=0.35)
    fig.suptitle('MCMC 采样诊断', fontsize=14, fontweight='bold')

    for j, (name, tv, col) in enumerate(
            zip(param_names, true_vals, cols)):
        chain = all_samples[:, j]
        post  = post_samp[:, j]

        # ── 链图 ──────────────────────────────────────────
        ax_trace = fig.add_subplot(gs[0, j])
        ax_trace.plot(chain, alpha=0.55, lw=0.5, color=col)
        ax_trace.axvline(n_burnin, color='red', ls='--', lw=1.5,
                         label=f'预热结束 (n={n_burnin})')
        ax_trace.axhline(tv, color='orange', lw=2,
                         label=f'真实值 = {tv:.4f}')
        ax_trace.axhline(post.mean(), color='purple', lw=1.5,
                         ls=':', label=f'后验均值 = {post.mean():.4f}')
        ax_trace.set_xlabel('迭代步数', fontsize=10)
        ax_trace.set_ylabel(name, fontsize=10)
        ax_trace.set_title(f'{name}  — 链图', fontsize=11)
        ax_trace.legend(fontsize=8)
        ax_trace.grid(True, alpha=0.3)

        # ── 自相关函数 (ACF) ──────────────────────────────
        ax_acf = fig.add_subplot(gs[1, j])
        max_lag = min(80, len(post) // 5)
        acf_vals = _compute_acf(post, max_lag)
        ci_bound = 1.96 / np.sqrt(len(post))
        ax_acf.bar(range(len(acf_vals)), acf_vals,
                   color=col, alpha=0.7, width=1.0)
        ax_acf.axhline(0,         color='black', lw=0.8)
        ax_acf.axhline( ci_bound, color='red', ls='--', lw=1,
                        label='95% CI')
        ax_acf.axhline(-ci_bound, color='red', ls='--', lw=1)
        ax_acf.set_xlabel('延迟 (lag)', fontsize=10)
        ax_acf.set_ylabel('自相关系数', fontsize=10)
        ax_acf.set_title(f'{name}  — 自相关函数 (ACF)', fontsize=11)
        ax_acf.legend(fontsize=8)
        ax_acf.set_xlim([0, max_lag])
        ax_acf.grid(True, alpha=0.3)

        # ── 累积均值收敛图 ────────────────────────────────
        ax_cum = fig.add_subplot(gs[2, j])
        steps     = np.arange(1, len(post) + 1)
        cum_mean  = np.cumsum(post) / steps
        ax_cum.plot(steps, cum_mean, color=col, lw=1.5,
                    label='累积后验均值')
        ax_cum.axhline(tv, color='orange', lw=2,
                       label=f'真实值 = {tv:.4f}')
        ax_cum.fill_between(
            steps,
            cum_mean - 2 * post.std() / np.sqrt(steps),
            cum_mean + 2 * post.std() / np.sqrt(steps),
            alpha=0.2, color=col, label='±2σ/√n')
        ax_cum.set_xlabel('后验样本数', fontsize=10)
        ax_cum.set_ylabel(name, fontsize=10)
        ax_cum.set_title(f'{name}  — 累积均值收敛', fontsize=11)
        ax_cum.legend(fontsize=8)
        ax_cum.grid(True, alpha=0.3)

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  ✓ {save_path}")


def _compute_acf(x: np.ndarray, max_lag: int) -> np.ndarray:
    """计算自相关函数"""
    x   = x - x.mean()
    var = np.var(x)
    if var == 0:
        return np.zeros(max_lag + 1)
    acf = [1.0]
    for lag in range(1, max_lag + 1):
        cov = np.mean(x[:len(x)-lag] * x[lag:])
        acf.append(cov / var)
    return np.array(acf)


# ============================================================
# 3. 后验边缘分布图
# ============================================================

def plot_posterior_marginals(posterior_samples: np.ndarray,
                              data: dict,
                              save_path: str = 'posterior_marginals.png'):
    """
    绘制 η_c 和 η_t 的后验边缘分布（直方图 + KDE），
    标注真实值、后验均值、95% 可信区间。
    """
    print("\n  [可视化] 绘制后验边缘分布…")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('贝叶斯后验边缘分布', fontsize=14, fontweight='bold')

    param_list = [
        (posterior_samples[:, 0], data['eta_c_true'],
         'η_c  压气机等熵效率', 'steelblue'),
        (posterior_samples[:, 1], data['eta_t_true'],
         'η_t  涡轮等熵效率',   'seagreen'),
    ]

    stats_out = {}
    for ax, (samp, tv, label, col) in zip(axes, param_list):
        mean_v = samp.mean()
        std_v  = samp.std()
        ci     = np.percentile(samp, [2.5, 97.5])
        key    = label.split()[0]
        stats_out[key] = dict(mean=mean_v, std=std_v, ci=ci)

        # 直方图
        ax.hist(samp, bins=60, density=True,
                alpha=0.55, color=col,
                edgecolor='white', linewidth=0.4,
                label='后验样本直方图')
        # KDE
        kde   = gaussian_kde(samp, bw_method='silverman')
        x_arr = np.linspace(samp.min() * 0.995,
                             samp.max() * 1.005, 300)
        ax.plot(x_arr, kde(x_arr), color=col, lw=2.5, label='后验 KDE')

        # 标注
        ax.axvline(tv,     color='red',    lw=2.5, ls='-',
                   label=f'真实值  = {tv:.4f}')
        ax.axvline(mean_v, color='orange', lw=2.0, ls='--',
                   label=f'后验均值 = {mean_v:.4f}')
        ax.axvspan(ci[0], ci[1], alpha=0.18, color=col,
                   label=f'95% CI [{ci[0]:.4f}, {ci[1]:.4f}]')

        # 误差注释
        rel_err = abs(mean_v - tv) / tv * 100
        ax.text(0.04, 0.95,
                f'相对误差：{rel_err:.3f}%\n'
                f'后验标准差：{std_v:.5f}',
                transform=ax.transAxes, fontsize=9,
                va='top', bbox=dict(boxstyle='round',
                                    facecolor='wheat', alpha=0.8))

        ax.set_xlabel(label, fontsize=12)
        ax.set_ylabel('概率密度', fontsize=12)
        ax.set_title(f'{label}  后验分布', fontsize=12)
        ax.legend(fontsize=8.5, loc='upper left')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  ✓ {save_path}")
    return stats_out


# ============================================================
# 4. 二维联合后验等高线图
# ============================================================

def plot_posterior_contour(posterior_samples: np.ndarray,
                            data: dict,
                            save_path: str = 'posterior_contour.png'):
    """
    绘制 (η_c, η_t) 二维联合后验分布等高线图
    """
    print("\n  [可视化] 绘制二维联合后验等高线…")

    eta_c_s = posterior_samples[:, 0]
    eta_t_s = posterior_samples[:, 1]
    tv_c    = data['eta_c_true']
    tv_t    = data['eta_t_true']

    # KDE 二维估计
    xy  = np.vstack([eta_c_s, eta_t_s])
    kde = gaussian_kde(xy, bw_method='scott')

    nc  = np.linspace(eta_c_s.min() * 0.997, eta_c_s.max() * 1.003, 100)
    nt  = np.linspace(eta_t_s.min() * 0.997, eta_t_s.max() * 1.003, 100)
    NC, NT = np.meshgrid(nc, nt)
    Z = kde(np.vstack([NC.ravel(), NT.ravel()])).reshape(NC.shape)

    fig, ax = plt.subplots(figsize=(8, 7))

    cf = ax.contourf(NC, NT, Z, levels=20, cmap='Blues', alpha=0.85)
    plt.colorbar(cf, ax=ax, label='后验概率密度')
    ax.contour(NC, NT, Z, levels=8,
               colors='navy', alpha=0.5, linewidths=0.8)

    # 散点（稀疏）
    idx = np.random.choice(len(eta_c_s),
                            size=min(800, len(eta_c_s)), replace=False)
    ax.scatter(eta_c_s[idx], eta_t_s[idx],
               alpha=0.08, s=4, color='steelblue', zorder=2)

    # 真实值 & 后验均值
    ax.plot(tv_c, tv_t, 'r*', ms=18, zorder=10,
            label=f'真实值 ({tv_c:.4f}, {tv_t:.4f})')
    ax.plot(eta_c_s.mean(), eta_t_s.mean(), 'y^', ms=14,
            markeredgecolor='darkorange', markeredgewidth=1.5,
            zorder=10,
            label=f'后验均值 ({eta_c_s.mean():.4f}, '
                  f'{eta_t_s.mean():.4f})')

    # 先验范围
    rect = plt.Rectangle((0.60, 0.60), 0.38, 0.38,
                          fill=False, edgecolor='gray',
                          ls='--', lw=1.5,
                          label='均匀先验范围 [0.60, 0.98]²')
    ax.add_patch(rect)

    ax.set_xlabel('η_c  压气机等熵效率', fontsize=13)
    ax.set_ylabel('η_t  涡轮等熵效率',   fontsize=13)
    ax.set_title('贝叶斯后验联合分布\n(η_c, η_t)',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=9.5, loc='lower right')
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  ✓ {save_path}")


# ============================================================
# 5. 模型拟合对比图
# ============================================================

def plot_model_fit(posterior_samples: np.ndarray,
                   data: dict,
                   engine: TurbojetEngine,
                   save_path: str = 'model_fit.png'):
    """
    用后验均值（MAP近似）预测性能，与观测数据对比。
    同时绘制后验预测区间（PPD）。
    """
    print("\n  [可视化] 绘制模型拟合对比图…")

    eta_c_map = posterior_samples[:, 0].mean()
    eta_t_map = posterior_samples[:, 1].mean()
    N  = data['N']

    # 后验均值预测
    F_pred   = np.zeros(N)
    SFC_pred = np.zeros(N)
    for i in range(N):
        F, SFC, ok = engine.compute_performance(
            data['pi_c'][i], data['T3'][i], eta_c_map, eta_t_map)
        F_pred[i]   = F   if ok else np.nan
        SFC_pred[i] = SFC if ok else np.nan

    # 后验预测区间（从后验样本中随机抽取部分，计算预测分布）
    n_ppd = min(300, len(posterior_samples))
    ppd_idx = np.random.choice(len(posterior_samples), n_ppd, replace=False)
    F_ppd   = np.zeros((n_ppd, N))
    SFC_ppd = np.zeros((n_ppd, N))
    for k, idx in enumerate(ppd_idx):
        ec, et = posterior_samples[idx]
        for i in range(N):
            F, SFC, ok = engine.compute_performance(
                data['pi_c'][i], data['T3'][i], ec, et)
            F_ppd[k, i]   = F   if ok else np.nan
            SFC_ppd[k, i] = SFC if ok else np.nan

    F_ppd_lo   = np.nanpercentile(F_ppd,    2.5, axis=0)
    F_ppd_hi   = np.nanpercentile(F_ppd,   97.5, axis=0)
    SFC_ppd_lo = np.nanpercentile(SFC_ppd,  2.5, axis=0) * 1e6
    SFC_ppd_hi = np.nanpercentile(SFC_ppd, 97.5, axis=0) * 1e6

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('后验预测 vs. 观测数据\n'
                 f'（后验均值：η_c={eta_c_map:.4f}, '
                 f'η_t={eta_t_map:.4f}）',
                 fontsize=13, fontweight='bold')

    for ax, y_obs, y_pred, y_lo, y_hi, xlabel, title, col in [
        (axes[0],
         data['F_obs'],    F_pred,         F_ppd_lo,    F_ppd_hi,
         'F_obs  [N/(kg/s)]', '比推力预测对比', 'steelblue'),
        (axes[1],
         data['SFC_obs']*1e6, SFC_pred*1e6, SFC_ppd_lo, SFC_ppd_hi,
         'SFC_obs  [mg/(N·s)]', '比油耗预测对比', 'seagreen'),
    ]:
        lim = [min(y_obs.min(), y_pred.min()) * 0.97,
               max(y_obs.max(), y_pred.max()) * 1.03]

        # 95% PPD 竖条
        for i in range(N):
            ax.plot([y_obs[i], y_obs[i]], [y_lo[i], y_hi[i]],
                    color=col, alpha=0.4, lw=3, zorder=2)

        ax.scatter(y_obs, y_pred, s=70, color=col,
                   edgecolors='white', lw=0.8,
                   zorder=5, label='后验均值预测')
        ax.plot(lim, lim, 'r--', lw=2, label='理想拟合线', zorder=3)
        ax.fill_between(lim,
                         [l * 0.99 for l in lim],
                         [l * 1.01 for l in lim],
                         alpha=0.15, color='red', label='±1% 误差带')

        # R² 计算
        mask = np.isfinite(y_pred)
        ss_res = np.sum((y_obs[mask] - y_pred[mask]) ** 2)
        ss_tot = np.sum((y_obs[mask] - y_obs[mask].mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

        ax.text(0.04, 0.94, f'R² = {r2:.6f}',
                transform=ax.transAxes, fontsize=11,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85))
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(xlabel.replace('obs', 'pred'), fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(lim); ax.set_ylim(lim)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  ✓ {save_path}")


# ============================================================
# 6. 残差分析图
# ============================================================

def plot_residuals(posterior_samples: np.ndarray,
                   data: dict,
                   engine: TurbojetEngine,
                   save_path: str = 'residuals.png'):
    """绘制标准化残差图，验证误差正态性假设"""
    print("\n  [可视化] 绘制残差分析图…")

    eta_c = posterior_samples[:, 0].mean()
    eta_t = posterior_samples[:, 1].mean()
    N = data['N']

    res_F   = np.zeros(N)
    res_SFC = np.zeros(N)
    for i in range(N):
        F, SFC, ok = engine.compute_performance(
            data['pi_c'][i], data['T3'][i], eta_c, eta_t)
        if ok:
            res_F[i]   = (data['F_obs'][i]   - F)   / data['sigma_F']
            res_SFC[i] = (data['SFC_obs'][i] - SFC) / data['sigma_SFC']

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle('标准化残差分析（验证误差正态性）',
                 fontsize=13, fontweight='bold')

    from scipy import stats as sc_stats
    for row, (res, label) in enumerate(
            [(res_F, '比推力'), (res_SFC, '比油耗')]):

        # 残差散点图
        ax1 = axes[row, 0]
        ax1.scatter(range(N), res, s=60, alpha=0.8, color=_COLORS[row])
        ax1.axhline(0, color='red', lw=1.5, ls='--')
        ax1.axhline( 2, color='orange', lw=1, ls=':')
        ax1.axhline(-2, color='orange', lw=1, ls=':')
        ax1.set_xlabel('观测点编号', fontsize=10)
        ax1.set_ylabel('标准化残差', fontsize=10)
        ax1.set_title(f'{label} — 标准化残差', fontsize=11)
        ax1.grid(True, alpha=0.3)

        # Q-Q 正态图
        ax2 = axes[row, 1]
        (osm, osr), (slope, intercept, r) = sc_stats.probplot(res)
        ax2.plot(osm, osr, 'o', color=_COLORS[row], alpha=0.8, ms=6)
        ax2.plot(osm, slope * np.array(osm) + intercept,
                 'r-', lw=2, label=f'正态参考线 (R={r:.4f})')
        ax2.set_xlabel('理论正态分位数', fontsize=10)
        ax2.set_ylabel('样本分位数', fontsize=10)
        ax2.set_title(f'{label} — Q-Q 正态图', fontsize=11)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  ✓ {save_path}")


# ============================================================
# 7. 综合摘要打印
# ============================================================

def print_inversion_summary(posterior_samples: np.ndarray,
                             data: dict,
                             info: dict):
    """格式化打印贝叶斯反演结果摘要"""
    eta_c_s = posterior_samples[:, 0]
    eta_t_s = posterior_samples[:, 1]

    print("\n" + "=" * 60)
    print("          贝叶斯反演结果摘要")
    print("=" * 60)
    print(f"  {'':15s} {'真实值':>10} {'后验均值':>10} "
          f"{'后验标准差':>12} {'95% CI':>24} {'相对误差':>10}")
    print("  " + "-" * 75)

    for name, samp, tv in [
        ('η_c (压气机效率)', eta_c_s, data['eta_c_true']),
        ('η_t (涡轮效率)',   eta_t_s, data['eta_t_true']),
    ]:
        mu  = samp.mean()
        std = samp.std()
        ci  = np.percentile(samp, [2.5, 97.5])
        err = abs(mu - tv) / tv * 100
        print(f"  {name:<15s} {tv:>10.4f} {mu:>10.4f} "
              f"{std:>12.5f} "
              f"[{ci[0]:.4f}, {ci[1]:.4f}] "
              f"{err:>9.3f}%")

    print("=" * 60)
    print(f"  总采样步数：{info['n_samples']:,}")
    print(f"  预热步数：  {info['n_burnin']:,}")
    print(f"  有效后验样本：{info['n_samples']-info['n_burnin']:,}")
    print(f"  MCMC 接受率：{info['acceptance_rate']:.4f}")
    print("=" * 60)
