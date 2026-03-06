"""
主程序入口
基于虚拟试验数据的涡喷发动机关键效率参数贝叶斯反演方法

毕业设计：基于虚拟试验数据的涡喷发动机关键效率参数贝叶斯反演方法研究

完整流程：
  ┌─ 步骤 1 ──────────────────────────────────────────────┐
  │  热力循环计算  →  绘制性能图（比推力、比油耗 vs. π_c, T3）│
  │  通过性能图选出符合原型机的工作参数范围                   │
  └────────────────────────────────────────────────────────┘
        ↓
  ┌─ 步骤 2 ──────────────────────────────────────────────┐
  │  在选定工作点处生成虚拟试验数据                          │
  │  y_obs = y_model(θ_true, x) + ε,  ε ~ N(0, σ²)       │
  └────────────────────────────────────────────────────────┘
        ↓
  ┌─ 步骤 3 ──────────────────────────────────────────────┐
  │  构建贝叶斯反演模型                                     │
  │  P(θ|D) ∝ P(D|θ) · P(θ)                              │
  │  对数似然（第七节）：                                    │
  │  ln L ∝ -∑[(F_i-F_model,i)²/2σ_F²                    │
  │            + (SFC_i-SFC_model,i)²/2σ_SFC²]            │
  └────────────────────────────────────────────────────────┘
        ↓
  ┌─ 步骤 4 ──────────────────────────────────────────────┐
  │  MCMC 采样 → 后验分布 P(η_c, η_t | D)                  │
  └────────────────────────────────────────────────────────┘
        ↓
  ┌─ 步骤 5 ──────────────────────────────────────────────┐
  │  后验分析：均值、标准差、95% CI、收敛诊断                 │
  │  可视化：链图、ACF、边缘分布、等高线、拟合对比             │
  └────────────────────────────────────────────────────────┘
"""

import os
import sys
import time
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')    # 非交互式后端（无显示器环境）

# ── 本地模块 ─────────────────────────────────────────────────────────────
from turbojet_engine import TurbojetEngine
from virtual_data    import (plot_performance_maps,
                              plot_optimal_region,
                              generate_virtual_data,
                              plot_observations)
from bayesian_model  import log_posterior
from mcmc_sampler    import MHSampler, MultiChainMH
from visualization   import (plot_likelihood_surface,
                              plot_mcmc_diagnostics,
                              plot_posterior_marginals,
                              plot_posterior_contour,
                              plot_model_fit,
                              plot_residuals,
                              print_inversion_summary)


# ============================================================
# 命令行参数
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='涡喷发动机效率参数贝叶斯反演')
    parser.add_argument('--eta-c-true', type=float, default=0.85,
                        help='压气机效率真实值（默认 0.85）')
    parser.add_argument('--eta-t-true', type=float, default=0.88,
                        help='涡轮效率真实值（默认 0.88）')
    parser.add_argument('--noise',  type=float, default=0.01,
                        help='测量噪声水平（占均值百分比，默认 1%%）')
    parser.add_argument('--n-samples', type=int, default=60000,
                        help='MCMC 总采样步数（默认 60000）')
    parser.add_argument('--n-burnin',  type=int, default=15000,
                        help='预热（burn-in）步数（默认 15000）')
    parser.add_argument('--step',   type=float, default=0.012,
                        help='MH 初始步长（默认 0.012）')
    parser.add_argument('--prior', type=str, default='uniform',
                        choices=['uniform', 'gaussian'],
                        help='先验类型（默认 uniform）')
    parser.add_argument('--multi-chain', action='store_true',
                        help='启用多链采样（Gelman-Rubin 诊断）')
    parser.add_argument('--n-chains', type=int, default=4,
                        help='多链数量（默认 4）')
    parser.add_argument('--outdir', type=str, default='.',
                        help='输出目录（默认当前目录）')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子（默认 42）')
    return parser.parse_args()


# ============================================================
# 辅助：保存路径
# ============================================================

def _path(outdir: str, fname: str) -> str:
    os.makedirs(outdir, exist_ok=True)
    return os.path.join(outdir, fname)


# ============================================================
# 主流程
# ============================================================

def main():
    args = parse_args()
    t_start = time.time()

    print("=" * 60)
    print("  基于虚拟试验数据的涡喷发动机关键效率参数贝叶斯反演")
    print("  Bayesian Inversion of Key Efficiency Parameters")
    print("  for Turbojet Engine — Virtual Experimental Data")
    print("=" * 60)
    print(f"  真实参数：η_c = {args.eta_c_true:.4f}, "
          f"η_t = {args.eta_t_true:.4f}")
    print(f"  噪声水平：{args.noise*100:.1f}%")
    print(f"  先验类型：{args.prior}")
    print(f"  MCMC 步数：{args.n_samples:,}（预热 {args.n_burnin:,}）")
    print(f"  输出目录：{args.outdir}/")
    print("=" * 60)

    # ──────────────────────────────────────────────────────
    # 初始化发动机模型
    # ──────────────────────────────────────────────────────
    engine = TurbojetEngine()

    # 验证循环计算（冒烟测试）
    F, SFC, ok = engine.compute_performance(
        8.0, 1300.0, args.eta_c_true, args.eta_t_true)
    assert ok, "发动机模型自检失败！"
    print(f"\n  [自检] π_c=8, T3=1300K → "
          f"F_sp={F:.2f} N/(kg/s), "
          f"SFC={SFC*1e6:.4f} mg/(N·s)  ✓")

    # ──────────────────────────────────────────────────────
    # 步骤 1：性能图（选优工作参数）
    # ──────────────────────────────────────────────────────
    plot_performance_maps(engine, args.eta_c_true, args.eta_t_true,
                          _path(args.outdir, 'performance_maps.png'))
    plot_optimal_region(engine, args.eta_c_true, args.eta_t_true,
                        _path(args.outdir, 'optimal_region.png'))

    # ──────────────────────────────────────────────────────
    # 步骤 2：生成虚拟观测数据
    # ──────────────────────────────────────────────────────
    data = generate_virtual_data(
        engine,
        eta_c_true=args.eta_c_true,
        eta_t_true=args.eta_t_true,
        noise_pct=args.noise,
        seed=args.seed,
    )
    plot_observations(data, _path(args.outdir, 'observations.png'))

    # ──────────────────────────────────────────────────────
    # 步骤 3：似然函数曲面（可视化参数辨识度）
    # ──────────────────────────────────────────────────────
    plot_likelihood_surface(
        data, engine,
        _path(args.outdir, 'likelihood_surface.png'))

    # ──────────────────────────────────────────────────────
    # 步骤 4：MCMC 采样求解后验分布
    # ──────────────────────────────────────────────────────
    # 先验参数
    prior_kwargs = {}
    if args.prior == 'gaussian':
        prior_kwargs = dict(
            mu_c=args.eta_c_true * 0.97,   # 偏置 3% 模拟不完整先验知识
            sigma_c=0.06,
            mu_t=args.eta_t_true * 0.97,
            sigma_t=0.06,
        )

    if args.multi_chain:
        # ── 多链并行 ──────────────────────────────────────
        mc = MultiChainMH(log_posterior, data, engine,
                          args.prior, prior_kwargs)
        posterior_samples, mc_info = mc.sample(
            n_chains=args.n_chains,
            n_samples=args.n_samples,
            n_burnin=args.n_burnin,
            step_size=args.step,
            seed_base=args.seed,
            verbose=True,
        )
        # 用第一条链的诊断信息
        chain_info = mc_info['infos'][0]
        chain_info['n_samples'] = args.n_samples
        chain_info['n_burnin']  = args.n_burnin
        all_samples_for_diag    = mc_info['chains'][0]['all_samples']

    else:
        # ── 单链 MH ───────────────────────────────────────
        sampler = MHSampler(log_posterior, data, engine,
                            args.prior, prior_kwargs)

        theta_init = np.array([0.80, 0.82])   # 初始猜测（偏离真实值）

        posterior_samples, chain_info = sampler.sample(
            theta_init,
            n_samples         = args.n_samples,
            n_burnin          = args.n_burnin,
            step_size         = args.step,
            target_acceptance = 0.234,
            seed              = args.seed,
            verbose           = True,
        )
        all_samples_for_diag = chain_info['all_samples']

    # 保存原始样本
    np.save(_path(args.outdir, 'posterior_samples.npy'), posterior_samples)
    np.save(_path(args.outdir, 'all_samples.npy'),       all_samples_for_diag)
    print(f"\n  ✓ 后验样本已保存 → {args.outdir}/posterior_samples.npy")

    # ──────────────────────────────────────────────────────
    # 步骤 5：结果分析与可视化
    # ──────────────────────────────────────────────────────
    print("\n" + "─" * 55)
    print("【步骤 5】结果分析与可视化")
    print("─" * 55)

    # MCMC 诊断
    plot_mcmc_diagnostics(
        all_samples_for_diag, chain_info, data,
        _path(args.outdir, 'mcmc_diagnostics.png'))

    # 后验边缘分布
    stats = plot_posterior_marginals(
        posterior_samples, data,
        _path(args.outdir, 'posterior_marginals.png'))

    # 二维联合后验
    plot_posterior_contour(
        posterior_samples, data,
        _path(args.outdir, 'posterior_contour.png'))

    # 模型拟合对比
    plot_model_fit(
        posterior_samples, data, engine,
        _path(args.outdir, 'model_fit.png'))

    # 残差分析
    plot_residuals(
        posterior_samples, data, engine,
        _path(args.outdir, 'residuals.png'))

    # 结果摘要
    print_inversion_summary(posterior_samples, data, chain_info)

    # ──────────────────────────────────────────────────────
    # 完成
    # ──────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    print(f"\n  总耗时：{elapsed:.1f} 秒")
    print("\n  输出文件列表：")
    for fname in [
        'performance_maps.png',
        'optimal_region.png',
        'observations.png',
        'likelihood_surface.png',
        'mcmc_diagnostics.png',
        'posterior_marginals.png',
        'posterior_contour.png',
        'model_fit.png',
        'residuals.png',
        'posterior_samples.npy',
        'all_samples.npy',
    ]:
        fpath = _path(args.outdir, fname)
        exists = '✓' if os.path.exists(fpath) else '✗'
        print(f"    [{exists}] {fname}")

    print("\n" + "=" * 60)
    print("  贝叶斯反演完成！")
    print("=" * 60)


# ============================================================
# 辅助：快速验证运行（无图形输出，用于 CI 测试）
# ============================================================

def quick_test():
    """
    快速冒烟测试：
      - 发动机模型正确性
      - 似然函数值域
      - MCMC 短链可运行
    """
    print("── 快速冒烟测试 ──")
    engine = TurbojetEngine()

    # 发动机计算
    F, SFC, ok = engine.compute_performance(8.0, 1300.0, 0.85, 0.88)
    assert ok and F > 0 and SFC > 0, "发动机计算失败"
    print(f"  发动机模型  OK  F={F:.2f}, SFC×1e6={SFC*1e6:.4f}")

    # 数据生成
    data = generate_virtual_data(engine, 0.85, 0.88,
                                  noise_pct=0.01, seed=0)
    assert data['N'] > 0
    print(f"  数据生成    OK  N={data['N']}")

    # 似然函数
    from bayesian_model import log_likelihood
    ll = log_likelihood([0.85, 0.88], data, engine)
    assert np.isfinite(ll), f"似然函数返回非有限值: {ll}"
    ll_bad = log_likelihood([0.3, 0.3], data, engine)
    assert ll > ll_bad or not np.isfinite(ll_bad)
    print(f"  似然函数    OK  ll(true)={ll:.2f}")

    # 短链 MCMC
    sampler = MHSampler(log_posterior, data, engine)
    samples, info = sampler.sample(
        np.array([0.80, 0.82]),
        n_samples=200, n_burnin=50, step_size=0.01, verbose=False)
    assert samples.shape == (150, 2)
    print(f"  MCMC 采样   OK  samples.shape={samples.shape}")

    print("── 所有测试通过 ✓ ──\n")
    return True


# ============================================================

if __name__ == '__main__':
    # 若以 --test 调用则运行快速测试，否则运行完整主程序
    if '--test' in sys.argv:
        quick_test()
    else:
        main()
