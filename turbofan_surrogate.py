"""
涡扇发动机简化代理前向模型
Turbofan Engine Simplified Surrogate Forward Model

对应 Mathcad 贝叶斯反演原型"第一节：代理前向模型定义"的 Python 实现。

模型参数 θ = (π_k, T_g, m)：
    π_k : 压气机总压比 (-)
    T_g : 涡轮进口温度 (K)
    m   : 涵道比 (-)

模型输出 y = (R_ud, C_ud)：
    R_ud : 比推力 (N·s/kg)
    C_ud : 比油耗 (kg/(N·h))

代理模型解析表达式（满足工程趋势，不要求真实物理精度）：
    R_ud(π_k, T_g, m) = 500 · (T_g/1000)^0.8 · π_k^0.25 / (1 + 0.25·m)
    C_ud(π_k, T_g, m) = 0.06 · (T_g/1200)^1.1 / (π_k^0.15 · (1 + 0.05·m))

物理趋势说明：
    ✓ R_ud 随 T_g 增大而单调增大（指数 0.8）
    ✓ R_ud 随 π_k 呈缓增型非线性变化（指数 0.25）
    ✓ R_ud 随 m 增大而减小（高涵道比稀释效应）
    ✓ C_ud 与 R_ud 趋势不完全相同（T_g 正向影响 C_ud，m 影响系数不同）
    ✓ 模型连续、平滑，支持全向量化参数扫描

后续替换说明：
    将本模块中的 R_ud / C_ud 解析式替换为真实热力循环公式，
    grid_bayes.py 中的贝叶斯推断框架无需任何修改即可使用。
"""

import numpy as np


class TurbofanSurrogate:
    """
    涡扇发动机简化代理前向模型

    参考设计点（用于验证）：
        π_k = 30,  T_g = 1500 K,  m = 8
        → R_ud ≈ 539.6 N·s/kg
        → C_ud ≈ 0.03290 kg/(N·h)
    """

    def __init__(self):
        # ── 代理模型标定系数 ──────────────────────────────────────────
        # 比推力公式系数
        self.A_R      = 500.0    # 比例系数        [N·s/kg]
        self.alpha_T  = 0.8      # T_g 指数（R_ud 对 T_g 灵敏度）
        self.alpha_pi = 0.25     # π_k 指数（R_ud 对 π_k 灵敏度，非线性）
        self.beta_m   = 0.25     # m 影响系数（分母项）

        # 比油耗公式系数
        self.A_C      = 0.06     # 比例系数        [kg/(N·h)]
        self.T_C_ref  = 1200.0   # 参考温度        [K]
        self.gamma_T  = 1.1      # T_g 指数（C_ud 对 T_g 灵敏度，略高）
        self.gamma_pi = 0.15     # π_k 指数（分母，增压比提升经济性）
        self.delta_m  = 0.05     # m 影响系数（分母项，涵道比影响较弱）

    # ------------------------------------------------------------------
    # 核心前向模型（支持标量 & NumPy 数组，完全向量化）
    # ------------------------------------------------------------------

    def compute_R_ud(self, pi_k, T_g, m):
        """
        计算比推力

        数学表达式（Mathcad 第一节，式 R_ud）：
            R_ud = 500 · (T_g/1000)^0.8 · π_k^0.25 / (1 + 0.25·m)

        Parameters
        ----------
        pi_k : float or ndarray — 压气机总压比
        T_g  : float or ndarray — 涡轮进口温度 [K]
        m    : float or ndarray — 涵道比

        Returns
        -------
        R_ud : float or ndarray — 比推力 [N·s/kg]
        """
        return (self.A_R
                * (T_g / 1000.0) ** self.alpha_T
                * pi_k ** self.alpha_pi
                / (1.0 + self.beta_m * m))

    def compute_C_ud(self, pi_k, T_g, m):
        """
        计算比油耗

        数学表达式（Mathcad 第一节，式 C_ud）：
            C_ud = 0.06 · (T_g/1200)^1.1 / (π_k^0.15 · (1 + 0.05·m))

        Parameters
        ----------
        pi_k : float or ndarray — 压气机总压比
        T_g  : float or ndarray — 涡轮进口温度 [K]
        m    : float or ndarray — 涵道比

        Returns
        -------
        C_ud : float or ndarray — 比油耗 [kg/(N·h)]
        """
        return (self.A_C
                * (T_g / self.T_C_ref) ** self.gamma_T
                / (pi_k ** self.gamma_pi * (1.0 + self.delta_m * m)))

    def compute_performance(self, pi_k, T_g, m):
        """
        单点计算（带有效性检查）

        Returns
        -------
        R_ud  : float
        C_ud  : float
        valid : bool
        """
        try:
            R = float(self.compute_R_ud(pi_k, T_g, m))
            C = float(self.compute_C_ud(pi_k, T_g, m))
            if R > 0 and C > 0 and np.isfinite(R) and np.isfinite(C):
                return R, C, True
        except Exception:
            pass
        return None, None, False

    def compute_grid(self, pi_grid, T_grid, m_grid):
        """
        在三维参数网格上批量计算性能量（完全向量化，无 Python 循环）

        对应 Mathcad 模板"六、网格后验计算"中的前向计算步骤。

        Parameters
        ----------
        pi_grid : ndarray, shape (Nπ,) — 压气机总压比网格
        T_grid  : ndarray, shape (NT,) — 涡轮进口温度网格 [K]
        m_grid  : ndarray, shape (Nm,) — 涵道比网格

        Returns
        -------
        R_3d : ndarray, shape (Nπ, NT, Nm) — 比推力三维数组
        C_3d : ndarray, shape (Nπ, NT, Nm) — 比油耗三维数组
        """
        # 构建 indexing='ij' 的 meshgrid，保证维度顺序为 (Nπ, NT, Nm)
        PI, TG, MG = np.meshgrid(pi_grid, T_grid, m_grid, indexing='ij')
        R_3d = self.compute_R_ud(PI, TG, MG)
        C_3d = self.compute_C_ud(PI, TG, MG)
        return R_3d, C_3d

    # ------------------------------------------------------------------
    # 趋势验证
    # ------------------------------------------------------------------

    def verify_trends(self):
        """
        验证代理模型的物理趋势，打印参数扫描结果。
        对应 Mathcad 模板"验证检查清单"第1条。
        """
        pi_ref, T_ref, m_ref = 30.0, 1500.0, 8.0
        R_ref = self.compute_R_ud(pi_ref, T_ref, m_ref)
        C_ref = self.compute_C_ud(pi_ref, T_ref, m_ref)

        print("=" * 55)
        print("  代理前向模型趋势验证")
        print("=" * 55)
        print(f"  设计参考点：π_k = {pi_ref:.0f},  T_g = {T_ref:.0f} K,  m = {m_ref:.0f}")
        print(f"  R_ud_ref = {R_ref:.4f}  N·s/kg    (Mathcad 验证值 ≈ 539.6)")
        print(f"  C_ud_ref = {C_ref:.6f} kg/(N·h)  (Mathcad 验证值 ≈ 0.032904)")
        print()

        print("  ── T_g 影响（π_k, m 固定在参考点）：")
        for T in [1200, 1350, 1500, 1650, 1800]:
            R = self.compute_R_ud(pi_ref, T, m_ref)
            C = self.compute_C_ud(pi_ref, T, m_ref)
            arrow = " ↑" if R > R_ref else (" ↓" if R < R_ref else "  ")
            print(f"    T_g = {T:4d} K → R_ud = {R:7.2f}{arrow},  C_ud = {C:.6f}")
        print("  结论：T_g↑ → R_ud↑ ✓，C_ud↑ ✓（两者同向但速率不同）")

        print()
        print("  ── π_k 影响（T_g, m 固定在参考点）：")
        for pi in [15, 20, 25, 30, 35, 40, 45]:
            R = self.compute_R_ud(pi, T_ref, m_ref)
            C = self.compute_C_ud(pi, T_ref, m_ref)
            print(f"    π_k = {pi:2d} → R_ud = {R:7.2f},  C_ud = {C:.6f}")
        print("  结论：π_k↑ → R_ud↑（次线性），C_ud↓ ✓（增压比提升经济性）")

        print()
        print("  ── m 影响（π_k, T_g 固定在参考点）：")
        for m_val in [4, 6, 8, 10, 12]:
            R = self.compute_R_ud(pi_ref, T_ref, m_val)
            C = self.compute_C_ud(pi_ref, T_ref, m_val)
            print(f"    m = {m_val:2d} → R_ud = {R:7.2f},  C_ud = {C:.6f}")
        print("  结论：m↑ → R_ud↓ ✓（高涵道比稀释），C_ud↓（略，效率提升）")
        print("=" * 55)

        return R_ref, C_ref


# ── 单独运行：趋势验证 ────────────────────────────────────────────────

if __name__ == '__main__':
    model = TurbofanSurrogate()
    model.verify_trends()
