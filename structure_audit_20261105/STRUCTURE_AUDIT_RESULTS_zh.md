# 双侧全局结构相似性审计：结果报告

生成目录：`/data/structure_audit_20261105/`
软件：US-align 20260328，`-mm 1 -ter 1`（多链复合物全局对齐，对齐第一模型的全部链）
对象：S1131（112 个 WT 复合物）、M1101（32 个 WT 复合物，含 5 个 HM）
失败：**0 / 6712 对**（另有 144 次自身对齐取长度，全部成功）

> 配套预检见 `PRECHECK_REPORT_zh.md`。所有数字来自实际 US-align 计算，未把失败或无法判定记录当作零暴露。

---

## 1. 方法（可复现）

1. 对每个数据集的全部独特 WT 复合物做**全对全**全局对齐：
   `USalign wt_i.pdb wt_j.pdb -mm 1 -ter 1`。
2. 每次对齐记录：
   - **双向归一化 TM-score**：`TM_i`（按结构 i 长度归一化）、`TM_j`（按结构 j 长度归一化）；
   - 对齐长度、RMSD、序列一致性；
   - **双向 coverage**：`aligned_len / len_i`、`aligned_len / len_j`。
3. 由全对全矩阵，对每个外层折取“每个评估复合物 → 训练复合物”的最佳匹配：
   - `best_eval_tm` = 训练复合物中对 `TM_eval`（按评估复合物长度归一化）取最大；
   - `bilateral` = 该对 `min(TM_eval, TM_train)` 的最大值（双侧都高才算高）；
   - 区分“**含相同复合物**”（训练里出现同一个 PDB，TM=1 平凡暴露）与“**排除相同复合物**”（novel 暴露）。
4. 阈值取 TM ≥ 0.5 与 TM ≥ 0.7（沿用本仓库既有的 0.70 讨论线）。
5. `RCSV/PCSV/ICSV` 用 `train_fold_k / val_fold_k`；`AGCV` 用 `fold_k_outer_train / fold_k_outer_eval`。

“双侧”在此实现为：整复合物全局对齐的**双向归一化**（TM 分别按两侧结构长度归一化）+ **双向 coverage**，并给出按行/按归一化方向展开的**双侧配对结构相似性矩阵**。本审计是**全局复合物**相似性，不声称界面/接触残基局部相似性。

## 2. 主结果（Supplementary Table S1 候选）

`supplementary_table_S1_structure_exposure.csv`，按“数据集 × 划分”对 25 个 fold×seed 单元取均值。

| 数据集 | 划分 | 评估复合物与训练相同（均值） | 最佳 TM≥0.7（含相同） | 最佳 TM≥0.7（排除相同） | 平均最佳 TM_eval | 平均双侧 min |
|---|---|---|---|---|---|---|
| S1131 | RCSV | 0.776 | 0.979 | 0.661 | 0.988 | 0.986 |
| S1131 | PCSV | 0.000 | 0.725 | 0.725 | 0.814 | 0.794 |
| S1131 | ICSV | 0.000 | 0.014 | 0.014 | 0.370 | 0.315 |
| M1101 | RCSV | 1.000 | 1.000 | 0.698 | 1.000 | 1.000 |
| M1101 | PCSV | 0.000 | 0.704 | 0.704 | 0.809 | 0.773 |
| M1101 | ICSV | 0.000 | 0.000 | 0.000 | 0.416 | 0.353 |
| M1101 | AGCV | 0.000 | 0.579 | 0.579 | 0.719 | 0.684 |

阈值 0.5（排除相同复合物）的评估复合物比例：
S1131 RCSV 0.692 / PCSV 0.747 / ICSV 0.039；
M1101 RCSV 0.885 / PCSV 0.869 / ICSV 0.126 / AGCV 0.763。

按**记录加权**（排除相同复合物、TM≥0.7）：S1131 RCSV 0.781、PCSV 0.764、ICSV 0.011；
M1101 RCSV 0.783、PCSV 0.733、ICSV 0.000、AGCV 0.612。

## 3. 解读

1. **RCSV 是随机划分**：约 78%（S1131）/ 100%（M1101）的评估复合物其同一 PDB 本身就出现在训练集里，整复合物 TM≈1。即便排除同复合物，仍有约 66%（S1131）/ 70%（M1101）的评估复合物能找到 TM≥0.7 的训练复合物。RCSV 的高性能不能解读为跨复合物泛化。

2. **PCSV（按伙伴划分）仍存在强结构暴露**：评估与训练复合物完全不重名（shared=0），但约 70–73% 的评估复合物仍有 TM≥0.7 的训练近邻。抽查的高匹配对都是真实同源复合物，例如 `1SGE↔1SGN`、`1Y33↔1Y1K`、`3BTG↔3BTD`、`1B2S↔1B3S`、`1CSO↔1CT4`、`1R0R↔2SIC`、`1TM3↔1TM4`。说明**按伙伴身份切分并未切断结构同源性**。

3. **ICSV（按序列相似性聚类）基本切断了全局结构暴露**：TM≥0.7 仅 1.4%（S1131）、0%（M1101），平均最佳 TM 降到约 0.37–0.42。序列聚类在结构层面是有效的（对全局骨架）。

4. **AGCV（抗原组留出）不是“无结构相似”**：评估复合物里约 58% 仍有 TM≥0.7 的训练近邻，被记录一加权仍有约 61%。典型例子是 `1KTZ ↔ HM_1KTZ`（TM 0.99，二者受体相同、仅 TGF-β 配体不同，按 README 被拆成两个抗原组）以及 `1T83 ↔ 3WJJ`（TM 0.95）。因此“实体级/抗原组留出”**不等于**“无结构同源性”，与既有 README 的声明一致。

5. **HM 条目**：5 个 HM 与其基础模板高度一致（模板存在，TM 见 §5）。HM 应作为独立条目参与，不得与基础 PDB 合并或删除前缀。

## 4. 重要限制（务必与 S1 一起引用）

- **全局 TM 包含抗体骨架**：M1101 多为抗体-抗原复合物，不同复合物共享 Ig 折叠，整复合物 TM 会被抗体骨架抬高。因此 M1101 的 PCSV/AGCV 暴露**部分来自共同抗体骨架**，不能直接等同于“抗原表位/界面相似”。界面级相似性、突变邻域相似性本次未计算。
- **AGCV 折的评估复合物数很小**（种子 0：每折 1、6、7、8、10 个复合物），复合物级比例方差很大（SD 达 0.31）；记录加权比例更稳定，建议正文优先引用记录级数字并注明折内复合物数。
- **五个种子不是独立划分**：M1101 ICSV 只有 1 套外层划分；AGCV 只有 4 套（种子 42 与 1234 相同）。25 个 fold×seed 单元不能当作 25 次独立重抽样；表中 SD 只描述单元间离散，不是显著性。
- **MUT 结构未参与**：审计只用 WT；MUT 仅用于映射核验。
- **HM 与基础模板的高相似是预期现象**（建模模板），不是外部同源泄漏。

## 5. HM 比较记录（`exposure/hm_template_summary.csv`）

| HM | 模板 | TM（按 HM 归一化） | TM（按模板归一化） | coverage(HM) | coverage(模板) | 最佳非模板基础 PDB（TM 按 HM） |
|---|---|---|---|---|---|---|
| HM_1KTZ | 1KTZ | 0.993 | 0.977 | 1.000 | 0.984 | 2NZ9 (0.402) |
| HM_1YY9 | 1YY9 | 1.000 | 0.409 | 1.000 | 0.409 | 3BN9 (0.645) |
| HM_2NYY | 2NYY | 0.984 | 1.000 | 0.984 | 1.000 | 2NZ9 (0.979) |
| HM_2NZ9 | 2NZ9 | 0.984 | 1.000 | 0.984 | 1.000 | 2NYY (0.979) |
| HM_3BN9 | 3BN9 | 1.000 | 1.000 | 1.000 | 1.000 | 2NYY (0.707) |

注：`HM_1YY9` 对 `1YY9` 按模板归一化仅 0.409（模板只被覆盖约 41%），提示 HM 与基础条目的链/残基覆盖范围不同，引用 `1YY9` 相关结构结论时需单独说明；`HM_2NYY` 与 `HM_2NZ9` 互相约 0.98，显示这组 HM 之间存在近乎重复的结构。

## 6. 输出文件索引

| 文件 | 内容 |
|---|---|
| `supplementary_table_S1_structure_exposure.csv` | **S1 汇总表**（数据集×划分，25 单元均值/SD） |
| `exposure/per_fold_exposure.csv` | 175 个 fold×seed 单元的逐折暴露统计 |
| `exposure/per_eval_complex_best_match.csv` | 3679 个“评估复合物→最佳训练复合物”记录 |
| `exposure/{ds}_matrix_tm_normalized_by_row.csv` | 双侧配对 TM 矩阵（按行结构归一化） |
| `exposure/{ds}_matrix_coverage_normalized_by_row.csv` | 双向 coverage 矩阵 |
| `pairwise/{ds}_pairwise.csv` | 逐对 TM（双向）、coverage（双向）、RMSD、对齐长度、序列一致性 |
| `pairwise/{ds}_complex_lengths.csv` | 每个复合物 US-align 口径长度与链名 |
| `exposure/hm_vs_base_all.csv` | 每个 HM 对全部基础 PDB 的逐对记录 |
| `exposure/hm_template_summary.csv` | HM 模板比较摘要 |
| `exposure/failed_or_undetermined.csv` | 失败/无法判定记录（本次为空） |
| `manifests/`、`precheck/` | 结构清单、链映射、划分完整性（见预检报告） |
| `scripts/`、`logs/` | 可复现脚本与运行日志。预览热图、暴露率条形图和 `scripts/plot_figures.py` 已删除 |

数值矩阵见 `exposure/{ds}_matrix_tm_normalized_by_row.csv`。

## 7. 复现命令

```bash
cd /data/structure_audit_20261105
python3 scripts/precheck.py
python3 scripts/run_pairwise.py --dataset M1101 --workers 32
python3 scripts/run_pairwise.py --dataset S1131 --workers 64
python3 scripts/fold_exposure.py
python3 scripts/aux_tables.py
```

环境与软件来源见 `usalign_bin/software_provenance.json`；二进制 SHA256 `dc6624938b7800a4bded5a7bde837b84ca04e4c14c7006ee0bf8a8fb83236408`。
