# 实验结果快照

两个阶段实验的汇总结果。详细的分析与结论见 `docs/` 下的两份记录:

- `docs/phase1-window-criterion.md` —— 叶子形态判据与点查代价
- `docs/phase2-range-queries.md` —— 范围查询

## 文件

### `logs_prod/` —— 生产路径(智能体在岗 + 无插桩 + 普通 `get()`)

由 `benchmark/prod_bench.cpp` 产生,`scripts/sweep_prod.sh` 驱动。每个 (数据集, 窗口门限)
一个日志,一次 build 带出点查 + 5 档范围。**这是唯一可以直接引用的性能数据**——上面那些 pilot
汇总表都是插桩构建测的,只能当消融数据。分析与结论见 `docs/phase3-production-e2e.md`。

当前覆盖(11 个日志):

| 数据集 | off | 4 | 16 | 1e9 |
|---|---|---|---|---|
| `uden.data` | ✓ | ✓ | ✓ | ✓ |
| `face.data` | ✓ | ✓ | ✓ | ✓ |
| `local_skew.data` | ✓ | ✓ | **点查 only** | 未跑 |

`local_skew.data_w16.log` 只有点查:范围查询在哈希叶上每次要扫满整片叶(该数据集按大小加权的叶是
**50,323 键 ≈ 1.24 MB**),该档的完整测量需要约 2 小时,被手动中止。`mixed.data` 完全未跑。

日志里每行:`prod:` 是结构/内存/构建耗时(其中 `structure=policy` 表示 DARE + TSMDP 在岗),
`point:` 与 `range:` 是延迟。**注意每条延迟都有 1~5% 的运行间噪声,跨条目差异小于 ~5% 时不应解读
为真实差异。**

### 其余文件

| 文件 | 内容 |
|---|---|
| `pilot_results.csv` | **点查**扫描。每次运行一行:数据集、门限、结构(是否固定)、叶数、合格叶分母、有序叶占比、`mem_ratio`、延迟 min/median/max、`errors` |
| `pilot_range.csv` | **范围**扫描。每次运行一行:选择性 `s`、区间键数、查询数、结构、有序叶占比、`mem_ratio`、平均结果数、延迟 min/median/max、`ns_per_result`、`errors` |
| `logs_range/` | 范围扫描每次运行的完整 stdout,含**叶大小分布**与**窗口(p99 预测误差)分布**——机制分析(按大小加权的叶大小等)的数据来源 |
| `pilot_range_early.csv` | 早期一批 face 运行。当时用 `LEAF_WINDOW_THRESHOLD=16`,只转换了 26% 的叶,因此测到的是"全哈希 vs 1/4 有序"的混合体(见阶段二文档 §5.1)。保留作为方法教训的原始记录 |
| `baseline_regression_check.csv` | 关闭判据、走原策略结构(root=417)的 uden 20M 基线,用于与早期 pilot 记录做回归对照:结构逐项相同(root/叶数/内节点跳数),内存差 0.22% 恰好等于诊断字段的 8 字节/叶 |

## 关键设置

- n = **20,000,000** 键(`<double key, double value>`,数据本身 320 MB)
- 结构钉死:`FIXED_CONF=256,16`、`TARGET_LEAF=1024`、`TARGET_FANOUT=16`
  —— 绕开 DARE 与 TSMDP 的结构决策,使叶子形态成为唯一自变量
- 叶子形态:哈希叶(基线)/ 有序叶(`LEAF_WINDOW_THRESHOLD` 生效;范围实验用 `1e9` 以转换全部合格叶)
- 查询集:固定种子,**确定性**;每轮切成 5 个 pass,丢弃首轮(预热),报各轮的 min/median/max
- 范围负载:分位数构造,`[lo,hi]` 覆盖数据集的固定比例 `s`
- 正确性:点查逐条比对返回的 value;范围查询对暴力 oracle(`lower_bound`/`upper_bound`)比对
  结果大小(每查询)+ 多重集逐元素(抽样)。两阶段的全部运行 `errors=0`

**延迟一律取 min-of-passes**,它是干扰的单侧最小估计;median 更接近"平均"但含干扰。
比跨运行噪声见两份文档的对应小节。

## 重新生成

```bash
cd Release && cmake . && make pilot_leaf pilot_range -j4

../scripts/sweep_pilot.sh                      # 点查扫描(20M × 4 数据集 × 5 门限)
EMIT=200000000 DS=face.data ../scripts/sweep_range.sh /tmp/f.csv   # 范围扫描
```

两个脚本都写成"可重跑同一配置"的形式:查询集种子固定,所以**重复执行可直接用来估计测量噪声**。

## 未包含

- 数据集本身(`data/`,约 6.6 GB,gitingore;生成方式见论文与 `index/include/Parameter.h` 的路径配置)
- 训练好的模型(`data/model/*.pt`)
- 逐叶原始转储 `leaf_stats_*.csv`(约 193 MB;需要时由 harness 重新生成)
- 构建产物
