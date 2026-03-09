# bishe1.0
## 基于虚拟试验数据的涡扇发动机关键参数贝叶斯反演方法研究

---

### 项目结构

#### A. 涡喷发动机真实热力循环 + MCMC 采样（主流程）

| 文件 | 说明 |
|---|---|
| `turbojet_engine.py` | 涡喷发动机热力循环计算模型（η_c, η_t） |
| `virtual_data.py` | 虚拟试验数据生成与性能图绘制 |
| `bayesian_model.py` | 对数似然、先验、后验函数 |
| `mcmc_sampler.py` | Metropolis-Hastings MCMC 采样器（自适应步长 + 多链） |
| `visualization.py` | 结果可视化（MCMC 诊断、边缘分布、等高线、拟合对比） |
| `main.py` | 完整主流程入口 |

**运行方式：**
```bash
python main.py
python main.py --eta-c-true 0.85 --eta-t-true 0.88 --noise 0.01
python main.py --multi-chain --n-chains 4
```

---

#### B. 涡扇发动机代理模型 + 网格后验（Mathcad 原型对应实现）

三参数循环反演：θ = (π_k, T_g, m) → y = (R_ud, C_ud)

| 文件 | 说明 |
|---|---|
| `turbofan_surrogate.py` | 简化代理前向模型（解析式，支持向量化） |
| `grid_bayes.py` | 网格离散贝叶斯反演（似然、后验、MAP、边缘分布） |
| `run_grid_demo.py` | 完整可行性验证演示脚本 |

**运行方式：**
```bash
python run_grid_demo.py                        # 默认参数（π_k=30, T_g=1500, m=8）
python run_grid_demo.py --outdir my_results    # 指定输出目录
python run_grid_demo.py --N-pi 30 --N-T 30 --N-m 25   # 更精细网格
python run_grid_demo.py --test                 # 快速冒烟测试
```

**输出图像：**
- `surrogate_model_check.png` — 代理模型参数敏感性分析（6子图）
- `posterior_marginals_1d.png` — 一维边缘后验 p(π_k|y), p(T_g|y), p(m|y)
- `posterior_contours_2d.png` — 二维联合后验等高线（3组）
- `inversion_summary.png` — 反演结果汇总（参数对比 + 数值表 + 后验切片）

---

### 代理前向模型

```
R_ud(π_k, T_g, m) = 500 · (T_g/1000)^0.8 · π_k^0.25 / (1 + 0.25·m)
C_ud(π_k, T_g, m) = 0.06 · (T_g/1200)^1.1 / (π_k^0.15 · (1 + 0.05·m))
```

设计参考点（π_k=30, T_g=1500K, m=8）：
- R_ud ≈ 539.5 N·s/kg
- C_ud ≈ 0.03289 kg/(N·h)

**后续扩展**：将 `turbofan_surrogate.py` 中的解析式替换为真实热力循环公式，`grid_bayes.py` 无需修改。
