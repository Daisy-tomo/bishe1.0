"""
基于网格的贝叶斯反演模块
Grid-based Bayesian Inversion for Turbofan Engine Cycle Parameters

对应 Mathcad 模板第三至九节（先验 → 似然 → 网格后验 → MAP → 边缘分布）的
完整 Python / NumPy 实现。

数学框架：
──────────────────────────────────────────────────────────────
  参数向量：θ = (π_k, T_g, m)
  观测量：  y_obs = (R_obs, C_obs)

  贝叶斯后验（对数形式）：
      ln P(θ|y) = ln P(y|θ) + ln P(θ) + const

  误差模型（独立高斯，对应 Mathcad 第四节对数似然）：
      ln L(θ) = -0.5 · [(R_pred - R_obs)²/σ_R²
                       + (C_pred - C_obs)²/σ_C²]

  均匀先验（Mathcad 第三节，网格内为常数 → 可省略）：
      ln P(θ) = 0  （网格内），-∞（越界，网格定义保证不发生）

  离散网格近似（对应 Mathcad 第六节）：
      后验矩阵 post[i, j, k] = exp(logL[i, j, k])

  一维边缘后验（对应 Mathcad 第七节）：
      p(π_k | y) ∝ Σ_{j,k} post[i, j, k] · ΔT · Δm
      p(T_g  | y) ∝ Σ_{i,k} post[i, j, k] · Δπ · Δm
      p(m    | y) ∝ Σ_{i,j} post[i, j, k] · Δπ · ΔT

  MAP 估计（对应 Mathcad 第八节）：
      θ_MAP = argmax p(θ | y_obs)
──────────────────────────────────────────────────────────────
"""

import numpy as np


class GridBayesianInversion:
    """
    三参数发动机循环参数网格贝叶斯反演器

    对应 Mathcad 模板的离散网格实现版本。
    全部后验计算使用 NumPy 广播/向量化，无 Python 层循环。

    使用流程：
        inv = GridBayesianInversion(surrogate_model)
        inv.compute_posterior(R_obs, C_obs)   # 计算后验
        p_pi, p_T, p_m = inv.marginal_1d()   # 一维边缘
        p_piT, p_pim, p_Tm = inv.marginal_2d() # 二维联合
        map_res = inv.compute_map()            # MAP 估计
        inv.print_summary(pi_true, T_true, m_true, ...)
    """

    def __init__(self, surrogate_model):
        """
        Parameters
        ----------
        surrogate_model : TurbofanSurrogate 或任何具有
                          compute_R_ud(pi, T, m) 和
                          compute_C_ud(pi, T, m) 方法的对象
        """
        self.model = surrogate_model

        # ── 均匀先验范围（Mathcad 第三节）──────────────────────────
        self.pi_k_min = 15.0;   self.pi_k_max = 45.0
        self.T_g_min  = 1200.0; self.T_g_max  = 1800.0
        self.m_min    = 4.0;    self.m_max    = 12.0

        # ── 网格分辨率（Mathcad 第四节 Nπ, NT, Nm）─────────────────
        self.N_pi = 25
        self.N_T  = 25
        self.N_m  = 20

        # ── 测量不确定度 σ（Mathcad 第五节）──────────────────────────
        self.sigma_R = 10.0       # [N·s/kg]
        self.sigma_C = 1.0e-3     # [kg/(N·h)]

        # ── 内部缓存（compute_posterior 调用后填充）────────────────
        self._pi_grid   = None    # shape (N_pi,)
        self._T_grid    = None    # shape (N_T,)
        self._m_grid    = None    # shape (N_m,)
        self._post_3d   = None    # shape (N_pi, N_T, N_m), 归一化后验
        self._logL_3d   = None    # shape (N_pi, N_T, N_m), 对数似然
        self._R_pred_3d = None    # shape (N_pi, N_T, N_m), 前向模型 R
        self._C_pred_3d = None    # shape (N_pi, N_T, N_m), 前向模型 C
        self._R_obs     = None
        self._C_obs     = None

    # ------------------------------------------------------------------
    # 网格步长属性（供外部使用，如归一化）
    # ------------------------------------------------------------------

    @property
    def d_pi(self):
        return (self.pi_k_max - self.pi_k_min) / max(self.N_pi - 1, 1)

    @property
    def d_T(self):
        return (self.T_g_max - self.T_g_min) / max(self.N_T - 1, 1)

    @property
    def d_m(self):
        return (self.m_max - self.m_min) / max(self.N_m - 1, 1)

    # ------------------------------------------------------------------
    # 网格向量（惰性初始化）
    # ------------------------------------------------------------------

    def _build_grid(self):
        """生成参数均匀网格（对应 Mathcad 第四节 π_grid, T_grid, m_grid）"""
        self._pi_grid = np.linspace(self.pi_k_min, self.pi_k_max, self.N_pi)
        self._T_grid  = np.linspace(self.T_g_min,  self.T_g_max,  self.N_T)
        self._m_grid  = np.linspace(self.m_min,    self.m_max,    self.N_m)

    @property
    def pi_grid(self):
        if self._pi_grid is None:
            self._build_grid()
        return self._pi_grid

    @property
    def T_grid(self):
        if self._T_grid is None:
            self._build_grid()
        return self._T_grid

    @property
    def m_grid(self):
        if self._m_grid is None:
            self._build_grid()
        return self._m_grid

    # ------------------------------------------------------------------
    # 核心：网格后验计算（完全向量化）
    # ------------------------------------------------------------------

    def compute_posterior(self, R_obs, C_obs):
        """
        在三维参数网格上计算归一化后验分布

        对应 Mathcad 模板第五节（似然函数）+ 第六节（网格后验计算）
        + 第七节第1步（后验归一化）。

        计算过程（全 NumPy 广播，无 Python 循环）：
            1. 构建 (Nπ, NT, Nm) meshgrid
            2. 批量计算前向模型 R_pred, C_pred
            3. 计算对数似然 logL[i,j,k]
            4. 数值稳定化：logL -= max(logL)
            5. 未归一化后验 = exp(logL_stable)
            6. 矩形数值积分归一化

        Parameters
        ----------
        R_obs : float — 观测比推力 [N·s/kg]
        C_obs : float — 观测比油耗 [kg/(N·h)]

        Returns
        -------
        post_3d : ndarray, shape (Nπ, NT, Nm) — 归一化后验概率密度
        """
        self._R_obs = float(R_obs)
        self._C_obs = float(C_obs)

        # ① 构建均匀网格
        self._build_grid()

        # ② 创建 (Nπ, NT, Nm) 三维网格（indexing='ij' → 第0轴是 π_k）
        PI, TG, MG = np.meshgrid(
            self._pi_grid, self._T_grid, self._m_grid, indexing='ij')

        # ③ 前向模型批量计算（向量化）
        R_pred = self.model.compute_R_ud(PI, TG, MG)
        C_pred = self.model.compute_C_ud(PI, TG, MG)
        self._R_pred_3d = R_pred
        self._C_pred_3d = C_pred

        # ④ 对数似然（Mathcad 第五节 logL 公式）
        #    logL = -0.5 · [(R_pred-R_obs)²/σ_R² + (C_pred-C_obs)²/σ_C²]
        logL = (-0.5 * (
            (R_pred - R_obs) ** 2 / self.sigma_R ** 2
            + (C_pred - C_obs) ** 2 / self.sigma_C ** 2
        ))
        self._logL_3d = logL

        # ⑤ 数值稳定化（防止 exp 下溢，减去最大值后归一化不影响后验形状）
        logL_stable = logL - np.max(logL)

        # ⑥ 未归一化后验（均匀先验在网格内为常数，省略先验因子）
        post_u = np.exp(logL_stable)

        # ⑦ 矩形积分归一化（Mathcad 第七节第1步）
        #    ∫∫∫ post_u dπ dT dm ≈ sum(post_u) · Δπ · ΔT · Δm
        Z = post_u.sum() * self.d_pi * self.d_T * self.d_m
        self._post_3d = post_u / Z

        return self._post_3d

    # ------------------------------------------------------------------
    # 一维边缘后验（Mathcad 第七节边缘分布 & 第八节图形数据）
    # ------------------------------------------------------------------

    def marginal_1d(self):
        """
        计算三个参数的归一化一维边缘后验

        数学表达式（Mathcad 第八节 p_π, p_T, p_m）：
            p(π_k|y) ∝ Σ_{j,k} post[i,j,k] · ΔT · Δm
            p(T_g|y) ∝ Σ_{i,k} post[i,j,k] · Δπ · Δm
            p(m|y)   ∝ Σ_{i,j} post[i,j,k] · Δπ · ΔT

        对应 Mathcad 中 marg_π[i]、marg_T[j]、marg_m[k] 向量。

        Returns
        -------
        p_pi : ndarray, shape (Nπ,) — ∫p dT dm（已归一化，∫p_π dπ ≈ 1）
        p_T  : ndarray, shape (NT,) — ∫p dπ dm
        p_m  : ndarray, shape (Nm,) — ∫p dπ dT
        """
        post = self._post_3d

        # 对其余两轴求和（等价于 Mathcad 程序块中的双重 for 循环）
        marg_pi = post.sum(axis=(1, 2)) * self.d_T  * self.d_m
        marg_T  = post.sum(axis=(0, 2)) * self.d_pi * self.d_m
        marg_m  = post.sum(axis=(0, 1)) * self.d_pi * self.d_T

        # 重新归一化（确保各自积分为 1）
        marg_pi /= (marg_pi.sum() * self.d_pi)
        marg_T  /= (marg_T.sum()  * self.d_T)
        marg_m  /= (marg_m.sum()  * self.d_m)

        return marg_pi, marg_T, marg_m

    # ------------------------------------------------------------------
    # 二维联合边缘后验（Mathcad 第八节图形输出 2）
    # ------------------------------------------------------------------

    def marginal_2d(self):
        """
        计算三组二维联合边缘后验（对第三个参数积分）

        数学表达式：
            p(π_k, T_g | y) ∝ Σ_k post[i,j,k] · Δm   → shape (Nπ, NT)
            p(π_k, m   | y) ∝ Σ_j post[i,j,k] · ΔT   → shape (Nπ, Nm)
            p(T_g, m   | y) ∝ Σ_i post[i,j,k] · Δπ   → shape (NT, Nm)

        对应 Mathcad 中 p_πT[i,j]、p_πm[i,k]、p_Tm[j,k] 矩阵。

        Returns
        -------
        p_piT : ndarray, shape (Nπ, NT)
        p_pim : ndarray, shape (Nπ, Nm)
        p_Tm  : ndarray, shape (NT, Nm)
        """
        post = self._post_3d

        p_piT = post.sum(axis=2) * self.d_m    # 对 m 轴积分
        p_pim = post.sum(axis=1) * self.d_T    # 对 T_g 轴积分
        p_Tm  = post.sum(axis=0) * self.d_pi   # 对 π_k 轴积分

        return p_piT, p_pim, p_Tm

    # ------------------------------------------------------------------
    # MAP 提取（Mathcad 第七节第2步）
    # ------------------------------------------------------------------

    def compute_map(self):
        """
        提取 MAP (Maximum A Posteriori) 估计

        两种 MAP：
        1. 联合 MAP：三维后验 post_3d 的全局最大值点
        2. 边缘 MAP：各一维边缘后验的各自最大值（更常用，更鲁棒）

        对应 Mathcad 第七节第2步 argmax_vec 程序函数。

        Returns
        -------
        dict 包含两种 MAP 的参数值及网格索引：
            pi_k_MAP, T_g_MAP, m_MAP  — 边缘 MAP（推荐使用）
            pi_k_MAP_joint, T_g_MAP_joint, m_MAP_joint — 联合 MAP
        """
        # ── 联合 MAP（三维全局最大值）───────────────────────────────
        idx_flat = np.argmax(self._post_3d)
        idx3 = np.unravel_index(idx_flat, self._post_3d.shape)

        # ── 边缘 MAP（每个参数的一维边缘后验最大值）────────────────
        p_pi, p_T, p_m = self.marginal_1d()
        idx_pi_m = int(np.argmax(p_pi))
        idx_T_m  = int(np.argmax(p_T))
        idx_m_m  = int(np.argmax(p_m))

        return {
            # 边缘 MAP（主结果）
            'pi_k_MAP': float(self._pi_grid[idx_pi_m]),
            'T_g_MAP':  float(self._T_grid[idx_T_m]),
            'm_MAP':    float(self._m_grid[idx_m_m]),
            'idx_pi':   idx_pi_m,
            'idx_T':    idx_T_m,
            'idx_m':    idx_m_m,
            # 联合 MAP（辅助参考）
            'pi_k_MAP_joint': float(self._pi_grid[idx3[0]]),
            'T_g_MAP_joint':  float(self._T_grid[idx3[1]]),
            'm_MAP_joint':    float(self._m_grid[idx3[2]]),
            'idx_joint':      idx3,
        }

    # ------------------------------------------------------------------
    # 误差分析（Mathcad 第七节第3步）
    # ------------------------------------------------------------------

    def compute_errors(self, pi_true, T_true, m_true):
        """
        计算 MAP 估计与真值的绝对误差和相对误差

        对应 Mathcad 第七节第3步：
            err_πk = π_k_MAP - π_k_true
            err_Tg = T_g_MAP - T_g_true
            err_m  = m_MAP   - m_true

        Parameters
        ----------
        pi_true, T_true, m_true : 真值参数

        Returns
        -------
        dict 包含绝对误差、相对误差（%），以及 MAP 处的性能量预测
        """
        map_res = self.compute_map()
        pi_MAP = map_res['pi_k_MAP']
        T_MAP  = map_res['T_g_MAP']
        m_MAP  = map_res['m_MAP']

        R_MAP = self.model.compute_R_ud(pi_MAP, T_MAP, m_MAP)
        C_MAP = self.model.compute_C_ud(pi_MAP, T_MAP, m_MAP)

        return {
            'pi_k_MAP': pi_MAP, 'T_g_MAP': T_MAP, 'm_MAP': m_MAP,
            'pi_k_true': pi_true, 'T_g_true': T_true, 'm_true': m_true,
            'err_pi_abs': pi_MAP - pi_true,
            'err_T_abs':  T_MAP  - T_true,
            'err_m_abs':  m_MAP  - m_true,
            'err_pi_pct': (pi_MAP - pi_true) / pi_true * 100.0,
            'err_T_pct':  (T_MAP  - T_true)  / T_true  * 100.0,
            'err_m_pct':  (m_MAP  - m_true)  / m_true  * 100.0,
            'R_MAP': float(R_MAP),
            'C_MAP': float(C_MAP),
        }

    # ------------------------------------------------------------------
    # 后验统计量（均值、标准差，用于不确定度量化）
    # ------------------------------------------------------------------

    def posterior_stats(self):
        """
        计算各参数的后验均值与标准差（对边缘后验数值积分）

        Returns
        -------
        dict: mu_pi, std_pi, mu_T, std_T, mu_m, std_m
        """
        p_pi, p_T, p_m = self.marginal_1d()

        def _mean_std(grid, p, dg):
            mu  = float(np.sum(grid * p) * dg)
            var = float(np.sum((grid - mu) ** 2 * p) * dg)
            return mu, float(np.sqrt(max(var, 0.0)))

        mu_pi, std_pi = _mean_std(self._pi_grid, p_pi, self.d_pi)
        mu_T,  std_T  = _mean_std(self._T_grid,  p_T,  self.d_T)
        mu_m,  std_m  = _mean_std(self._m_grid,  p_m,  self.d_m)

        return {
            'mu_pi': mu_pi, 'std_pi': std_pi,
            'mu_T':  mu_T,  'std_T':  std_T,
            'mu_m':  mu_m,  'std_m':  std_m,
        }

    # ------------------------------------------------------------------
    # 摘要打印（Mathcad 第八节结果对比表）
    # ------------------------------------------------------------------

    def print_summary(self, pi_true, T_true, m_true, R_true, C_true):
        """
        格式化打印反演结果摘要表

        对应 Mathcad 模板第八节"真值与 MAP 对比表"。
        """
        err   = self.compute_errors(pi_true, T_true, m_true)
        stats = self.posterior_stats()

        print("\n" + "=" * 70)
        print("       贝叶斯反演结果摘要  （代理模型 + 网格后验 + 均匀先验）")
        print("=" * 70)
        print(f"  {'参数':<12} {'真值':>8} {'后验均值':>10} "
              f"{'后验标准差':>12} {'MAP':>9} {'误差%':>9}")
        print("  " + "─" * 62)
        print(f"  {'π_k':<12} {pi_true:>8.2f} {stats['mu_pi']:>10.3f} "
              f"{stats['std_pi']:>12.4f} {err['pi_k_MAP']:>9.3f} "
              f"{err['err_pi_pct']:>+8.2f}%")
        print(f"  {'T_g [K]':<12} {T_true:>8.1f} {stats['mu_T']:>10.2f} "
              f"{stats['std_T']:>12.3f} {err['T_g_MAP']:>9.2f} "
              f"{err['err_T_pct']:>+8.2f}%")
        print(f"  {'m':<12} {m_true:>8.2f} {stats['mu_m']:>10.3f} "
              f"{stats['std_m']:>12.4f} {err['m_MAP']:>9.3f} "
              f"{err['err_m_pct']:>+8.2f}%")
        print()
        print(f"  {'性能量':<18} {'真值':>12} {'观测值':>12} {'MAP预测':>12}")
        print("  " + "─" * 58)
        print(f"  {'R_ud [N·s/kg]':<18} {R_true:>12.4f} "
              f"{self._R_obs:>12.4f} {err['R_MAP']:>12.4f}")
        print(f"  {'C_ud [×10⁻³]':<18} {C_true*1e3:>12.5f} "
              f"{self._C_obs*1e3:>12.5f} {err['C_MAP']*1e3:>12.5f}")
        print("  (C_ud 单位：kg/(N·h)，表中已乘 1000)")
        print()
        print(f"  网格规格：{self.N_pi} × {self.N_T} × {self.N_m}"
              f" = {self.N_pi * self.N_T * self.N_m:,} 个网格点")
        print(f"  σ_R = {self.sigma_R} N·s/kg,  "
              f"σ_C = {self.sigma_C:.2e} kg/(N·h)")
        print("=" * 70)
