"""
涡喷发动机循环计算模型
Turbojet Engine Thermodynamic Cycle Model

数学描述（对应"一、参数反演问题的数学描述"）：
    y = f(θ, x)
    y = [F_sp, SFC]  — 发动机性能参数（比推力，比油耗）
    θ = [η_c, η_t]   — 待反演参数（压气机效率，涡轮效率）
    x = [π_c, T3]    — 循环输入（增压比，涡轮入口温度）

循环站位定义：
    0 → 进气道入口（大气）
    1 → 压气机入口
    2 → 压气机出口 / 燃烧室入口
    3 → 燃烧室出口 / 涡轮入口
    4 → 涡轮出口 / 喷管入口
    5 → 喷管出口（排气）
"""

import numpy as np


class TurbojetEngine:
    """
    涡喷发动机理想热力循环计算模型

    假设：
    - 工质为完全气体（γ=1.4，Cp=1005 J/(kg·K)）
    - 进气道无损失（等熵）
    - 燃烧室无压力损失（等压燃烧）
    - 喷管完全膨胀（出口静压等于大气压）
    - 涡轮与压气机功率平衡（忽略机械损失与引气）
    - 地面静止工况（飞行速度 V0 = 0）
    """

    def __init__(self,
                 T0: float = 288.15,
                 P0: float = 101325.0,
                 gamma: float = 1.4,
                 Cp: float = 1005.0,
                 Hu: float = 43.0e6):
        """
        初始化发动机参数

        Parameters
        ----------
        T0    : 环境温度 [K]，默认 288.15 K（ISA 海平面）
        P0    : 环境压力 [Pa]，默认 101325 Pa
        gamma : 工质比热比 [-]，默认 1.4（空气）
        Cp    : 定压比热容 [J/(kg·K)]，默认 1005
        Hu    : 燃料低热值 [J/kg]，默认 43 MJ/kg（航空煤油）
        """
        self.T0    = T0
        self.P0    = P0
        self.gamma = gamma
        self.Cp    = Cp
        self.Hu    = Hu
        self.V0    = 0.0   # 飞行速度（地面静止）

    # ------------------------------------------------------------------
    # 核心循环计算
    # ------------------------------------------------------------------

    def compute_performance(self,
                            pi_c: float,
                            T3: float,
                            eta_c: float,
                            eta_t: float):
        """
        计算发动机单工况比推力与比油耗

        Parameters
        ----------
        pi_c  : 压气机增压比 π_c  [-]
        T3    : 涡轮入口总温  T3   [K]
        eta_c : 压气机等熵效率 η_c [-]，范围 (0, 1)
        eta_t : 涡轮等熵效率   η_t [-]，范围 (0, 1)

        Returns
        -------
        F_sp  : 比推力 F/ṁ  [N/(kg/s)]
        SFC   : 比油耗       [kg/(N·s)]
        valid : bool，本工况是否物理合理
        """
        g = self.gamma
        Cp = self.Cp
        T0 = self.T0
        P0 = self.P0

        # ── 参数合理性检查 ──────────────────────────────────────────
        if not (0.5 < eta_c < 1.0 and 0.5 < eta_t < 1.0):
            return None, None, False
        if pi_c <= 1.0 or T3 <= T0:
            return None, None, False

        try:
            # ── 进气道（0→1，静止无损失） ──────────────────────────
            T01 = T0
            P01 = P0

            # ── 压气机（1→2，绝热压缩） ────────────────────────────
            # 等熵压比温比
            tau_c_s = pi_c ** ((g - 1.0) / g)
            # 实际出口总温（含等熵效率）
            T02 = T01 * (1.0 + (tau_c_s - 1.0) / eta_c)
            P02 = P01 * pi_c
            # 单位质量压气机耗功
            W_c = Cp * (T02 - T01)

            # ── 燃烧室（2→3，等压燃烧） ────────────────────────────
            T03 = T3
            P03 = P02   # 忽略压力损失
            if T03 <= T02:          # 需要燃烧加热
                return None, None, False
            # 燃油-空气比（能量守恒）
            f = Cp * (T03 - T02) / self.Hu
            if f <= 0 or f > 0.08:  # 物理限制：f < ~0.06 典型值
                return None, None, False

            # ── 涡轮（3→4，绝热膨胀） ──────────────────────────────
            # 轴功平衡：涡轮功 = 压气机功（1+f≈1 近似，忽略燃油质量流量）
            # 涡轮实际出口总温
            T04 = T03 - W_c / Cp
            if T04 <= 0:
                return None, None, False

            # 涡轮实际温降与等熵温降
            delta_T_t_act = T03 - T04          # 实际温降
            delta_T_t_s   = delta_T_t_act / eta_t  # 等熵温降（含效率）
            T04s = T03 - delta_T_t_s           # 等熵出口总温

            if T04s <= 0 or T04s >= T03:
                return None, None, False

            # 涡轮膨胀比（由等熵温比推导）
            pi_t = (T03 / T04s) ** (g / (g - 1.0))
            # 涡轮出口总压
            P04 = P03 / pi_t

            # ── 喷管（4→5，等熵膨胀，完全膨胀 P5=P0） ─────────────
            if P04 <= P0:           # 无膨胀能力（欠膨胀检查）
                return None, None, False

            # 喷管出口静温
            T5 = T04 * (P0 / P04) ** ((g - 1.0) / g)
            if T5 <= 0:
                return None, None, False

            # 喷管出口速度（完全等熵膨胀）
            V_e = np.sqrt(2.0 * Cp * (T04 - T5))

            # ── 性能参数计算 ────────────────────────────────────────
            # 比推力 [N/(kg/s)]  (V0=0，地面静止)
            F_sp = V_e - self.V0
            if F_sp <= 0:
                return None, None, False

            # 比油耗 [kg/(N·s)]
            SFC = f / F_sp

            return F_sp, SFC, True

        except Exception:
            return None, None, False

    # ------------------------------------------------------------------
    # 批量计算（用于贝叶斯采样）
    # ------------------------------------------------------------------

    def compute_batch(self,
                      pi_c_arr: np.ndarray,
                      T3_arr: np.ndarray,
                      eta_c: float,
                      eta_t: float):
        """
        批量计算多工况性能参数

        Returns
        -------
        F_arr   : 比推力数组  [N/(kg/s)]
        SFC_arr : 比油耗数组  [kg/(N·s)]
        valid   : 有效性布尔数组
        """
        N = len(pi_c_arr)
        F_arr   = np.zeros(N)
        SFC_arr = np.zeros(N)
        valid   = np.zeros(N, dtype=bool)

        for i in range(N):
            F, SFC, ok = self.compute_performance(
                float(pi_c_arr[i]), float(T3_arr[i]), eta_c, eta_t
            )
            if ok:
                F_arr[i]   = F
                SFC_arr[i] = SFC
                valid[i]   = True

        return F_arr, SFC_arr, valid

    # ------------------------------------------------------------------
    # 热力循环参数查询（调试用）
    # ------------------------------------------------------------------

    def cycle_states(self, pi_c, T3, eta_c, eta_t):
        """返回各站位热力参数字典（调试/可视化用）"""
        g, Cp = self.gamma, self.Cp
        T01, P01 = self.T0, self.P0

        tau_c_s = pi_c ** ((g - 1.0) / g)
        T02 = T01 * (1.0 + (tau_c_s - 1.0) / eta_c)
        P02 = P01 * pi_c
        W_c = Cp * (T02 - T01)
        f   = Cp * (T3 - T02) / self.Hu

        T04 = T3 - W_c / Cp
        delta_T_t_s = (T3 - T04) / eta_t
        T04s = T3 - delta_T_t_s
        pi_t = (T3 / T04s) ** (g / (g - 1.0))
        P04  = P02 / pi_t

        T5 = T04 * (self.P0 / P04) ** ((g - 1.0) / g)
        V_e = np.sqrt(2.0 * Cp * (T04 - T5))

        return {
            'T01': T01, 'P01': P01,
            'T02': T02, 'P02': P02,
            'T03': T3,  'P03': P02,
            'T04': T04, 'P04': P04,
            'T5' : T5,  'P5' : self.P0,
            'V_e': V_e,
            'f'  : f,
            'W_c': W_c,
            'pi_t': pi_t,
            'F_sp': V_e,
            'SFC' : f / V_e if V_e > 0 else np.inf
        }
