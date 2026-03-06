"""
MCMC 采样器模块
Markov Chain Monte Carlo Sampler

用于求解贝叶斯后验分布 P(θ|D)。
实现了两种采样策略：

1. Metropolis-Hastings (MH)
   - 经典对称高斯提案分布
   - 自适应步长调整（AM-MCMC）
   - 基础可靠，易于理解与诊断

2. Delayed Rejection Adaptive Metropolis (DRAM)  [可选]
   - 拒绝时尝试第二个（更小步长的）提案
   - 对混合较差的后验有更好的鲁棒性

参考文献：
  Hastings W.K. (1970). Monte Carlo sampling methods using Markov chains
  Gelman A. et al. (2013). Bayesian Data Analysis
"""

import numpy as np
from typing import Callable, Tuple
from bayesian_model import log_posterior
from turbojet_engine import TurbojetEngine


# ============================================================
# 工具函数
# ============================================================

def _acceptance_rate(n_accepted: int, n_total: int) -> float:
    return n_accepted / n_total if n_total > 0 else 0.0


def _adapt_step(step: float, acc_rate: float,
                target: float = 0.234,
                scale: float = 0.9) -> float:
    """
    自适应步长调整。

    目标接受率：
      - 一维参数最优约 0.44
      - 高维（d→∞）最优约 0.234
    """
    if acc_rate < target:
        return step * scale
    return step / scale


# ============================================================
# Metropolis-Hastings 采样器（带自适应步长）
# ============================================================

class MHSampler:
    """
    自适应 Metropolis-Hastings MCMC 采样器

    算法流程（对应文档第八节 MCMC 求解后验分布）：
    ──────────────────────────────────────────────
    初始化  θ⁰ = theta_init
    对 t = 1, 2, ..., N：
        1. 提案：θ* ~ q(θ* | θ^{t-1}) = N(θ^{t-1}, Σ_proposal)
        2. 接受概率：α = min(1, P(θ*|D) / P(θ^{t-1}|D))
                      = min(1, exp(ln P(θ*|D) - ln P(θ^{t-1}|D)))
        3. u ~ Uniform[0, 1]
           若 u < α：θ^t = θ*   （接受）
           否则：    θ^t = θ^{t-1}（拒绝，链留在原位）
    预热期（burn-in）：丢弃前 n_burnin 个样本
    ──────────────────────────────────────────────
    """

    def __init__(self,
                 log_post_fn: Callable,
                 data: dict,
                 engine: TurbojetEngine,
                 prior_type: str = 'uniform',
                 prior_kwargs: dict = None):
        """
        Parameters
        ----------
        log_post_fn  : 对数后验函数（来自 bayesian_model）
        data         : 观测数据字典
        engine       : TurbojetEngine 实例
        prior_type   : 先验类型 'uniform' 或 'gaussian'
        prior_kwargs : 先验额外参数
        """
        self.log_post = log_post_fn
        self.data         = data
        self.engine       = engine
        self.prior_type   = prior_type
        self.prior_kwargs = prior_kwargs or {}

    def _eval_log_post(self, theta: np.ndarray) -> float:
        return self.log_post(
            theta, self.data, self.engine,
            self.prior_type, self.prior_kwargs
        )

    def sample(self,
               theta_init: np.ndarray,
               n_samples: int = 60000,
               n_burnin: int  = 15000,
               step_size: float = 0.01,
               adapt_interval: int = 200,
               adapt_burnin: int   = 10000,
               target_acceptance: float = 0.234,
               seed: int = 0,
               verbose: bool = True) -> Tuple[np.ndarray, dict]:
        """
        执行 MCMC 采样

        Parameters
        ----------
        theta_init        : 初始参数 [η_c, η_t]
        n_samples         : 总采样步数（含预热）
        n_burnin          : 预热（burn-in）步数
        step_size         : 初始提案步长（各维度相同的各向同性高斯）
        adapt_interval    : 步长自适应调整间隔
        adapt_burnin      : 自适应停止步数（之后步长固定）
        target_acceptance : 目标接受率
        seed              : 随机种子
        verbose           : 是否打印进度

        Returns
        -------
        samples  : shape (n_samples - n_burnin, d) 后验样本（去掉预热）
        info     : 诊断信息字典
        """
        rng = np.random.default_rng(seed)
        dim = len(theta_init)

        # 存储所有样本（含预热）
        all_samples = np.zeros((n_samples, dim))
        all_samples[0] = theta_init.copy()
        log_p_curr = self._eval_log_post(theta_init)

        n_accepted   = 0
        step_history = [step_size]

        if verbose:
            print("─" * 55)
            print("【步骤 3】Metropolis-Hastings MCMC 采样")
            print("─" * 55)
            print(f"  总采样步数：{n_samples:,}")
            print(f"  预热步数：  {n_burnin:,}")
            print(f"  后验样本数：{n_samples - n_burnin:,}")
            print(f"  初始步长：  {step_size:.4f}")
            print(f"  初始值：    η_c={theta_init[0]:.4f}, "
                  f"η_t={theta_init[1]:.4f}")
            print(f"  目标接受率：{target_acceptance:.3f}")
            print()
            milestones = {
                int(n_samples * p): f"{p*100:.0f}%"
                for p in [0.1, 0.2, 0.3, 0.4, 0.5,
                           0.6, 0.7, 0.8, 0.9, 1.0]
            }

        for t in range(1, n_samples):
            # ── 1. 生成提案（对称高斯） ───────────────────────────
            proposal = all_samples[t - 1] + rng.normal(
                0.0, step_size, dim)

            # ── 2. 计算接受概率（对数域，数值稳定） ───────────────
            log_p_prop = self._eval_log_post(proposal)
            log_alpha  = log_p_prop - log_p_curr

            # ── 3. MH 接受/拒绝准则 ──────────────────────────────
            if np.log(rng.uniform()) < log_alpha:
                all_samples[t] = proposal
                log_p_curr     = log_p_prop
                n_accepted    += 1
            else:
                all_samples[t] = all_samples[t - 1]

            # ── 4. 自适应步长（仅在预热期内调整） ─────────────────
            if t < adapt_burnin and t % adapt_interval == 0:
                acc = _acceptance_rate(n_accepted, t)
                step_size = _adapt_step(step_size, acc, target_acceptance)
                step_size = float(np.clip(step_size, 1e-5, 0.3))
                step_history.append(step_size)

            # ── 进度输出 ──────────────────────────────────────────
            if verbose and t in milestones:
                acc = _acceptance_rate(n_accepted, t)
                print(f"  [{milestones[t]:>4s}] step {t:>7,} | "
                      f"接受率 {acc:.3f} | 步长 {step_size:.5f} | "
                      f"当前 η_c={all_samples[t,0]:.4f} "
                      f"η_t={all_samples[t,1]:.4f}")

        final_acc = _acceptance_rate(n_accepted, n_samples)
        if verbose:
            print(f"\n  最终接受率：{final_acc:.4f}  "
                  f"（推荐范围 0.15~0.50）")
            if final_acc < 0.10:
                print("  ⚠ 接受率偏低，建议减小步长或检查先验范围")
            elif final_acc > 0.60:
                print("  ⚠ 接受率偏高，建议增大步长以提升探索效率")
            else:
                print("  ✓ 接受率正常")
            print("  ✓ MCMC 采样完成")

        # ── 去掉预热期 ────────────────────────────────────────────
        posterior_samples = all_samples[n_burnin:]

        info = dict(
            all_samples   = all_samples,
            n_accepted    = n_accepted,
            acceptance_rate = final_acc,
            step_history  = np.array(step_history),
            n_burnin      = n_burnin,
            n_samples     = n_samples,
        )
        return posterior_samples, info


# ============================================================
# 多链并行（可选，用于收敛诊断）
# ============================================================

class MultiChainMH:
    """
    多链 MH 采样器，用于 Gelman-Rubin 收敛诊断（R̂ 统计量）

    运行 n_chains 条独立链，从不同初始点出发。
    """

    def __init__(self, log_post_fn, data, engine,
                 prior_type='uniform', prior_kwargs=None):
        self.sampler = MHSampler(log_post_fn, data, engine,
                                 prior_type, prior_kwargs)

    def sample(self,
               n_chains: int = 4,
               n_samples: int = 40000,
               n_burnin: int  = 10000,
               step_size: float = 0.01,
               seed_base: int = 100,
               eta_c_bounds=(0.60, 0.98),
               eta_t_bounds=(0.60, 0.98),
               verbose: bool = True) -> Tuple[np.ndarray, dict]:
        """
        运行多条独立 MCMC 链

        初始点从先验范围内随机抽取，确保链的多样性。
        """
        print("─" * 55)
        print(f"【多链 MCMC】运行 {n_chains} 条独立链")
        print("─" * 55)

        rng = np.random.default_rng(seed_base)
        all_chains  = []
        all_infos   = []

        for k in range(n_chains):
            # 随机初始点（在先验范围内）
            theta_init = np.array([
                rng.uniform(*eta_c_bounds),
                rng.uniform(*eta_t_bounds),
            ])
            print(f"\n  ── 链 {k+1}/{n_chains} ──  "
                  f"初始值 η_c={theta_init[0]:.4f}, "
                  f"η_t={theta_init[1]:.4f}")

            samples, info = self.sampler.sample(
                theta_init,
                n_samples    = n_samples,
                n_burnin     = n_burnin,
                step_size    = step_size,
                seed         = seed_base + k,
                verbose      = verbose,
            )
            all_chains.append(samples)
            all_infos.append(info)

        # 合并所有链
        merged = np.concatenate(all_chains, axis=0)

        # Gelman-Rubin R̂ 统计量
        rhat = _gelman_rubin(all_chains)
        print(f"\n  Gelman-Rubin R̂：η_c = {rhat[0]:.4f}，"
              f"η_t = {rhat[1]:.4f}  （< 1.1 视为收敛）")
        if all(r < 1.1 for r in rhat):
            print("  ✓ 所有链已收敛")
        else:
            print("  ⚠ 部分链尚未充分收敛，建议增加采样步数")

        return merged, dict(chains=all_chains, infos=all_infos, rhat=rhat)


def _gelman_rubin(chains: list) -> np.ndarray:
    """
    计算 Gelman-Rubin 收敛诊断统计量 R̂

    R̂ < 1.1 表示链已充分收敛

    Parameters
    ----------
    chains : list of (n, d) arrays，每条链的后验样本

    Returns
    -------
    rhat : shape (d,) 每个参数的 R̂ 值
    """
    M = len(chains)             # 链数
    N = min(c.shape[0] for c in chains)  # 最短链长
    chains = [c[:N] for c in chains]
    chains_arr = np.array(chains)        # (M, N, d)
    d = chains_arr.shape[2]

    rhat = np.zeros(d)
    for j in range(d):
        # 链内均值与链间均值
        psi_j  = chains_arr[:, :, j]    # (M, N)
        psi_m  = psi_j.mean(axis=1)     # 各链均值 (M,)
        psi_bar = psi_m.mean()          # 总均值

        # 链间方差 B
        B = N / (M - 1) * np.sum((psi_m - psi_bar) ** 2)
        # 链内方差 W（各链方差的均值）
        s2_m = psi_j.var(axis=1, ddof=1)  # (M,)
        W = s2_m.mean()

        # 后验方差估计
        var_hat = (N - 1) / N * W + B / N
        rhat[j] = np.sqrt(var_hat / W) if W > 0 else np.inf

    return rhat
