"""
贝叶斯反演模型模块
Bayesian Inversion Model

严格对应"一、参数反演问题的数学描述"各章节：

【三】误差概率模型
    ε_F   ~ N(0, σ_F²)
    ε_SFC ~ N(0, σ_SFC²)

【四】单变量似然函数
    P(F_obs | θ)   = 1/√(2πσ_F²)  · exp(-(F_obs-F_model)²  / (2σ_F²))
    P(SFC_obs | θ) = 1/√(2πσ_SFC²)· exp(-(SFC_obs-SFC_model)²/(2σ_SFC²))

【五】联合似然函数（假设 F 误差与 SFC 误差统计独立）
    P(D | θ) = P(F_obs|θ) · P(SFC_obs|θ)
             = 1/(2πσ_F σ_SFC) · exp[-(F_obs-F_model)²/(2σ_F²)
                                      -(SFC_obs-SFC_model)²/(2σ_SFC²)]

【六】多组数据联合似然（N 组独立观测）
    P(D | θ) = ∏_{i=1}^{N} P(F_i, SFC_i | θ)

【七】对数似然函数（数值稳定，去掉常数项）
    ln P(D | θ) ∝ -∑_{i=1}^{N} [(F_i - F_{model,i})²/(2σ_F²)
                                 + (SFC_i - SFC_{model,i})²/(2σ_SFC²)]
    → 贝叶斯反演本质等价于加权最小二乘问题

【八】贝叶斯后验分布
    P(θ | D) ∝ P(D | θ) · P(θ)
    采用 MCMC 采样方法求解后验分布。
"""

import numpy as np
from turbojet_engine import TurbojetEngine


# ============================================================
# 【三、四、五、六、七】 似然函数
# ============================================================

def log_likelihood(theta: np.ndarray,
                   data: dict,
                   engine: TurbojetEngine) -> float:
    """
    多工况联合对数似然函数

    数学表达式（文档第七节）：
        ln L(θ|D) ∝ -∑_{i=1}^N [
            (F_i - F_{model,i})²  / (2 σ_F²)
          + (SFC_i - SFC_{model,i})²/ (2 σ_SFC²)
        ]

    Parameters
    ----------
    theta  : [η_c, η_t]
    data   : 观测数据字典（generate_virtual_data 的返回值）
    engine : TurbojetEngine 实例

    Returns
    -------
    float : 对数似然值（-∞ 表示参数无效）
    """
    eta_c, eta_t = float(theta[0]), float(theta[1])

    pi_c_arr  = data['pi_c']
    T3_arr    = data['T3']
    F_obs     = data['F_obs']
    SFC_obs   = data['SFC_obs']
    sigma_F   = data['sigma_F']
    sigma_SFC = data['sigma_SFC']
    N         = data['N']

    # 单点贡献的分母常数（避免重复计算）
    inv_2sigF2   = 1.0 / (2.0 * sigma_F   ** 2)
    inv_2sigSFC2 = 1.0 / (2.0 * sigma_SFC ** 2)

    log_L = 0.0
    for i in range(N):
        F_model, SFC_model, ok = engine.compute_performance(
            pi_c_arr[i], T3_arr[i], eta_c, eta_t
        )
        if not ok:
            return -np.inf  # 物理无效，概率为零

        # 各分量对数似然贡献（去掉常数 -N·ln(2πσ_Fσ_SFC)）
        log_L -= (F_obs[i]   - F_model)   ** 2 * inv_2sigF2
        log_L -= (SFC_obs[i] - SFC_model) ** 2 * inv_2sigSFC2

    return log_L


def likelihood(theta: np.ndarray,
               data: dict,
               engine: TurbojetEngine) -> float:
    """
    联合似然函数（非对数形式，用于可视化）

    P(D|θ) = exp(ln L(θ|D))
    """
    ll = log_likelihood(theta, data, engine)
    return np.exp(ll) if np.isfinite(ll) else 0.0


# ============================================================
# 【八】先验分布
# ============================================================

def log_prior(theta: np.ndarray,
              eta_c_bounds: tuple = (0.60, 0.98),
              eta_t_bounds: tuple = (0.60, 0.98)) -> float:
    """
    参数先验对数概率

    采用均匀先验（无信息先验），范围基于工程物理知识：
        η_c ∈ [0.60, 0.98]
        η_t ∈ [0.60, 0.98]

    P(θ) = 1 / [(0.98-0.60)²]  （归一化常数，后验推导时可忽略）

    当参数超出范围时，P(θ)=0 → ln P(θ)=-∞

    Parameters
    ----------
    theta          : [η_c, η_t]
    eta_c_bounds   : η_c 先验范围 (下界, 上界)
    eta_t_bounds   : η_t 先验范围 (下界, 上界)
    """
    eta_c, eta_t = float(theta[0]), float(theta[1])
    if (eta_c_bounds[0] <= eta_c <= eta_c_bounds[1] and
            eta_t_bounds[0] <= eta_t <= eta_t_bounds[1]):
        return 0.0     # ln(1/C) = -ln(C)，C 为常数，MAP 推导中可忽略
    return -np.inf


def log_prior_gaussian(theta: np.ndarray,
                       mu_c: float = 0.84, sigma_c: float = 0.05,
                       mu_t: float = 0.87, sigma_t: float = 0.05) -> float:
    """
    高斯先验（可选，当已知工程经验值时使用）

    P(η_c) ~ N(μ_c, σ_c²)
    P(η_t) ~ N(μ_t, σ_t²)

    参数独立，联合先验为乘积：
        ln P(θ) = ln P(η_c) + ln P(η_t)
    """
    eta_c, eta_t = float(theta[0]), float(theta[1])

    # 物理边界硬约束
    if not (0.5 < eta_c < 1.0 and 0.5 < eta_t < 1.0):
        return -np.inf

    lp = -0.5 * ((eta_c - mu_c) / sigma_c) ** 2
    lp -= 0.5 * ((eta_t - mu_t) / sigma_t) ** 2
    return lp


# ============================================================
# 【八】 对数后验分布
# ============================================================

def log_posterior(theta: np.ndarray,
                  data: dict,
                  engine: TurbojetEngine,
                  prior_type: str = 'uniform',
                  prior_kwargs: dict = None) -> float:
    """
    对数后验分布

    贝叶斯公式：
        ln P(θ|D) = ln P(D|θ) + ln P(θ) + const

    Parameters
    ----------
    theta        : [η_c, η_t]
    data         : 观测数据
    engine       : 发动机模型
    prior_type   : 'uniform'（均匀先验） 或 'gaussian'（高斯先验）
    prior_kwargs : 先验函数的额外参数字典
    """
    if prior_kwargs is None:
        prior_kwargs = {}

    # 先验
    if prior_type == 'uniform':
        lp = log_prior(theta, **prior_kwargs)
    elif prior_type == 'gaussian':
        lp = log_prior_gaussian(theta, **prior_kwargs)
    else:
        raise ValueError(f"未知先验类型：{prior_type}")

    if not np.isfinite(lp):
        return -np.inf

    # 似然
    ll = log_likelihood(theta, data, engine)
    if not np.isfinite(ll):
        return -np.inf

    return lp + ll


# ============================================================
# 似然函数曲面（可视化辅助）
# ============================================================

def compute_likelihood_surface(data: dict,
                                engine: TurbojetEngine,
                                eta_c_range: np.ndarray = None,
                                eta_t_range: np.ndarray = None) -> tuple:
    """
    在 (η_c, η_t) 网格上计算对数似然值，用于可视化曲面。

    Returns
    -------
    ETA_C, ETA_T : meshgrid
    LOG_L        : 对数似然矩阵（形状与网格相同）
    """
    if eta_c_range is None:
        eta_c_range = np.linspace(0.65, 0.97, 80)
    if eta_t_range is None:
        eta_t_range = np.linspace(0.65, 0.97, 80)

    ETA_C, ETA_T = np.meshgrid(eta_c_range, eta_t_range)
    LOG_L = np.full(ETA_C.shape, np.nan)

    total = ETA_C.size
    count = 0
    for i in range(ETA_C.shape[0]):
        for j in range(ETA_C.shape[1]):
            ll = log_likelihood(
                [ETA_C[i, j], ETA_T[i, j]], data, engine)
            if np.isfinite(ll):
                LOG_L[i, j] = ll
            count += 1
        if (i + 1) % 10 == 0:
            pct = 100 * count / total
            print(f"    似然曲面计算进度：{pct:.0f}%", end='\r')

    print("    似然曲面计算完成。         ")
    return ETA_C, ETA_T, LOG_L
