# 双侧配对结构暴露率核查报告（论文公式 4–8）

> 计算目录：`structure_audit_paired_20261105/`
> 旧整复合物分析目录（**未改动**）：`structure_audit_20261105/`
> US-align 版本 20260328，命令 `USalign <partner_i> <partner_j> -mm 1 -ter 1`
> 阈值 τ=0.70、c=0.80；主统计量 **record-weighted，含相同复合物匹配**。

本报告只提交数据与结论，**未修改论文正文**，也未触碰任何既有正式结果。

---

## 0. 执行摘要（先看结论）

| 数据集 | 划分 | record-weighted 暴露（含相同复合物，主口径） | record-weighted（排除相同复合物） | complex-weighted（含相同） |
|---|---|---|---|---|
| S1131 | RCSV | **0.992** | 0.460 | 0.972 |
| S1131 | PCSV | **0.447** | 0.447 | 0.644 |
| S1131 | ICSV | **0.000** | 0.000 | 0.000 |
| M1101 | RCSV | **1.000** | 0.628 | 1.000 |
| M1101 | PCSV | **0.577** | 0.577 | 0.607 |
| M1101 | ICSV | **0.000** | 0.000 | 0.000 |
| M1101 | AGCV | **0.330** | 0.330 | 0.358 |

（表中为 5 seed×5 fold=25 个 cell 的均值；pooled 值见 S1。）

核心结论：**按公式 4–8 严格实现后，PCSV 仍保留显著的跨 PDB 双侧结构近邻；ICSV 完全消除双侧配对暴露；AGCV 仍有明显结构近邻；RCSV 因允许相同 PDB 跨折而接近饱和（这是定义内预期，不是违规）。**

---

## 1. 新的双侧配对暴露率与旧整复合物指标的区别

**根本区别**：旧指标（`structure_audit_20261105/`）是**整复合物多链全局 TM-score**——对每对复合物只跑一次整复合物 US-align，按**整复合物总残基长度**归一化，`min(TM_eval, TM_train)` 是对**同一个整复合物比对的两个归一化方向**取小。它**没有**左右伙伴的概念。

新指标严格按公式 4–8：**分别抽取左右伙伴**，左只对左、右只对右各跑一次 US-align，每侧按**伙伴长度**归一化得到 `S_p = min(TM_{p,v}, TM_{p,t})`，再 `S_pair = min(S_left, S_right)`，并要求**同一个训练复合物 t** 两侧同时达标。

二者**不等价**，且**双向都不一致**。用旧目录 `pairwise/{ds}_pairwise.csv` 与新 `pairwise_{ds}_combined.csv` 交叉判定（`qc_diagnostics.py`）：

- **旧指标会假阳性（whole_only）**：整复合物 TM≥0.7 但配对 `S_pair` 很低。
  - 例（S1131）：`1SMF vs 3BTF`，整复合物 min TM=0.827，但 **S_left=0.115, S_right=0.997 → S_pair=0.115**。原因是整复合物比对把“右侧高相似”稀释进全局长度归一化，掩盖了左侧完全不同。
  - 例（M1101）：`1MLC vs 2JEL`，整复合物 0.780，`S_pair=0.276`；`1MHP vs 2JEL` 整复合物 0.730，`S_pair=0.293`。
  - S1131 共 **93** 对、M1101 共 **10** 对出现整复合物与配对判定相反。
- **旧指标也会假阴性（paired_only）**：如 M1101 `3BN9 vs 3NPS`，整复合物 0.680（<0.7 判不暴露），但 **S_left=0.921, S_right=0.991 → S_pair=0.921 且 C_pair=0.982，按公式 8 应为暴露**。

**量级差异**：以 S1131 PCSV 为例，旧整复合物“novel≥0.7 复合物比例”约 0.725，而新的严格双侧配对 record-weighted 仅 **0.447**（complex-weighted 0.644）。旧值偏高，因为它把整复合物单一全局 TM 当作“两侧都相似”的代理，而公式 8 要求**两侧各自**达标。因此 **不能**用旧 S1 支撑论文的 paired exposure 数字，两者口径不可互换。

---

## 2. PCSV 是否仍保留明显的跨 PDB 双侧结构相似性？—— **是的，显著保留**

PCSV 下没有相同 PDB 跨折（shared=0），暴露全部来自**不同 PDB** 的真实双侧结构近邻：

- S1131 PCSV：record-weighted **0.447**（排除相同=相同，因本就无相同）；complex-weighted **0.644**。
- M1101 PCSV：record-weighted **0.577**，complex-weighted **0.607**。

典型双侧强近邻（`S_pair≥0.7`，同 t）：

| 数据集 | 复合物对 | S_left | S_right | S_pair | C_pair |
|---|---|---|---|---|---|
| S1131 | 1KAC / 1P69 | 1.00 | 1.00 | 1.00 | 1.00 |
| S1131 | 3BTQ / 3BTT | 1.00 | 1.00 | 1.00 | 1.00 |
| S1131 | 1TM1 / 1Y34 | 1.00 | 1.00 | 1.00 | 1.00 |
| M1101 | 1BJ1 / 1CZ8 | 1.00 | 0.99 | 0.99 | 1.00 |
| M1101 | 1N8Z / 3BE1 | 0.95 | 0.97 | 0.95 | 0.98 |
| M1101 | 1DQJ / 1MLC | 0.93 | 0.98 | 0.93 | 0.98 |

S1131 全矩阵中 `S_pair≥0.7` 的复合物对共 **322** 对。结论：**PCSV 在 PDB 层面分离，但双侧重链/双抗原高度同源的跨条目结构暴露大量存在**，这正是论文中“PCSV 保留 paired structural similarity”的定量证据。

---

## 3. ICSV 是否显著降低双侧配对结构暴露？—— **是，降为 0**

- S1131 ICSV：record/complex-weighted **均为 0.000**。
- M1101 ICSV：**均为 0.000**。

在左右伙伴**分别**做序列聚类（论文 eq 3）后，训练/验证两侧不可能同时出现同簇的左右伙伴，因此不存在同一 t 两侧都达标的情形。这与“ICSV 消除跨折 partner 序列关系”的设计意图一致，且是**配对定义**下严格为 0，而非整复合物口径的“接近 0”。

---

## 4. AGCV 是否仍有明显结构近邻？—— **是**

M1101 AGCV（使用 outer_train/outer_eval）：record-weighted **0.330**，complex-weighted **0.358**。暴露主要来自抗原分组内的高度同源复合物（seed 0 的暴露复合物及其同一训练匹配）：

- `1KTZ ↔ HM_1KTZ`（S_pair≈0.94）、`1T83 ↔ 3WJJ`、`3BN9 ↔ HM_3BN9`、`3NPS ↔ HM_3BN9`、`3WJJ ↔ 1T83`。
- 说明即便按抗原分组滚动验证，**同一抗原/近缘抗原的双侧结构近邻仍跨折存在**，AGCV 并未消除结构暴露。

---

## 5. 五个 seed 的结果如何变化？

per_seed 的 record-weighted（含相同复合物）跨 seed 波动：

| 数据集/划分 | seed 最小值 | seed 最大值 | 极差 |
|---|---|---|---|
| S1131 RCSV | 0.989 | 0.995 | 0.005 |
| S1131 PCSV | 0.421 | 0.462 | 0.041 |
| S1131 ICSV | 0.000 | 0.000 | 0.000 |
| M1101 RCSV | 1.000 | 1.000 | 0.000 |
| M1101 PCSV | 0.563 | 0.629 | 0.065 |
| M1101 ICSV | 0.000 | 0.000 | 0.000 |
| M1101 AGCV | 0.350 | 0.350 | 0.000 |

- **RCSV、ICSV、AGCV 跨 seed 几乎不变**（RCSV 因允许相同 PDB，近乎 1；ICSV 恒 0；AGCV 因为按抗原分组、fold 组成稳定，5 seed 完全相同）。
- **PCSV 有 4–6 个百分点的小幅 seed 波动**，反映不同划分把特定复合物放训练/验证边界的差异，但总体结论（PCSV 仍有大量双侧暴露、远高于 ICSV）在所有 seed 下一致。

---

## 6. 记录加权与复合物加权是否一致？

**方向一致，但数值有系统差异**（这是数据本身“少数大复合物贡献很多记录”的结构所致）：

- S1131 PCSV：record-weighted 0.447 < complex-weighted 0.644 —— **未暴露的验证复合物平均承载的记录更多**（例如记录数很多的抗体类复合物往往不被结构近邻匹配），故按记录加权会拉低暴露率；complex-weighted 更高说明“暴露复合物（数量占比）较大，但它们记录较少”。二者同向（都显著>0、都远低于 RCSV、都远高于 ICSV）。
- M1101 PCSV：record 0.577 ≈ complex 0.607，接近。
- M1101 AGCV：record 0.330 ≈ complex 0.358，接近。
- RCSV：record≈complex≈饱和；ICSV：两者均 0。

结论：两种加权在**所有划分上判定方向一致**，不存在“一个说暴露、一个说完全不暴露”的矛盾；数值差异可解释为记录在复合物间的不均匀分配。论文主报 record-weighted 是稳妥的，但建议在补充材料同时给出 complex-weighted，避免“复合物数≠记录数”的误读（与论文既有观点一致）。

---

## 7. 是否存在影响可信度的结构映射/比对问题？

**已核查，无阻断性问题**，但如实记录以下事项：

1. **左右伙伴映射已交叉核对**：S1131 的 `R1–R6` 与 `Heavy+Light`、`L1–L3` 与 `Antigen` 链**逐条一致**（1131 条记录 `nR/L` 全部匹配；序列级抽查 R1==重链、L1==抗原成立）。M1101 以 `antigen_mapping.csv` 的 `antibody_chains/antigen_chains` 为准，30 个复合物与 CSV 列一致，**2 个复合物 `1KTZ`、`HM_1KTZ` 的 metadata 左右角色与 CSV 列相反**（CSV 把 `A` 记 Heavy、`B` 记 Antigen；metadata 认定 `B` 为抗体受体、`A` 为 TGF-β 配体）。已按用户要求**以 metadata 为准**，并在 `partner_chain_conflicts.csv` 中显式记录。此交换对配对结果影响有限（左右都用同源结构比较，且两者互为同源），但已透明标注。
2. **链抽取完整**：144 个复合物**全部**满足左右链集合不相交、并集==WT 全部 ATOM 链（`chain_partition_ok=yes` 100%），无缺失链。
3. **多链比对链匹配验证**：多链左伙伴（26 个，如 `1BJ1 H+L`、`3NPS B+C`）和多链右伙伴（3 个，`1BJ1/1CZ8/1DVF` 的 V+W / C+D）在 US-align 的 `Name of Structure:...:chain_list` 中链数与预期一致（如 `1BJ1.left.pdb:H:L`、`1BJ1.right.pdb:V:W`），说明 `-mm 1 -ter 1` 正确按多链处理。
4. **自匹配 QC**：全部 288 个（4 组）自比对 TM=1、coverage=1，无异常。
5. **左右矩阵独立**：`paired_complex_S_left_matrix.csv` 与 `paired_complex_S_right_matrix.csv` 分别由左右两套独立 US-align 结果构建；`paired_complex_similarity_matrix.csv` 为其 `min`。已确认左矩阵非对称结构中左右来源不同（非同一矩阵复制）。
6. **同 t 严格要求已程序化验证**：`qc_diagnostics.py` 独立重算 `min(S_left,S_right)` 与 combined 表中 `S_pair`，S1131 6216 对、M1101 496 对**全部零误差**；暴露判定要求同一 t 两侧同时 `S_pair≥0.7 & C_pair≥0.8`，代码层面不存在跨 t 拼接左右最佳。
7. **失败/未决**：`failed_or_undetermined.csv` 为空——无 US-align 失败、无 UNRESOLVED。全部比对成功（S1131 左右各 6216 + 含自比对；M1101 左右各 496）。UNRESOLVED 未计入 NOT_EXPOSED（分母用 resolved 数，本数据两者相等）。
8. **HM 独立对象**：`HM_1KTZ/HM_1YY9/HM_2NYY/HM_2NZ9/HM_3BN9` 全程作为独立结构对象，未与基础 PDB 合并。
9. **局部界面相似性**：未纳入（符合范围约定）。

唯一需要读者知晓的方法学事实：**`1KTZ/HM_1KTZ` 的左右角色在 metadata 与 CSV 列之间相反，本次以 metadata 为准**。对这两个复合物而言，metadata 认定 `B` 为抗体受体、`A` 为 TGF-β 配体（CSV 列恰好相反），按定义应当比较“抗体侧对同源受体”。需要说明：左右角色会决定“哪个链与对方哪个链做同侧比较”，因此该交换**确会改变** `1KTZ/HM_1KTZ` 与其他复合物的逐侧比对对象；本次严格采用 metadata 的生物学角色（而非 CSV 列名），这正是用户要求并与论文/Supplementary Methods 一致的做法。`1KTZ ↔ HM_1KTZ` 两者同时交换，其配对结果等价，`S_pair` 对这对自洽。

---

## 8. 与历史 45.3% / 56.4% 的关系（仅作参考，未强行对齐）

论文正文提到的 PCSV 配对结构暴露 **45.3%(S1131)、56.4%(M1101)** 是**历史参考值**。本次独立实现得到：

- S1131 PCSV：record-weighted **0.447**（≈0.45，接近历史 45.3%）。
- M1101 PCSV：record-weighted **0.577**（≈0.58，约合 57.7%，与历史 56.4% 同量级、略有差异）。

**没有为接近历史数值调整任何定义**。差异来源可解释为：M1101 以 metadata 角色为准（含 1KTZ 角色交换）、阈值严格 `C_pair≥0.80`（旧脚本无 coverage 门槛）、以及复合物集合/记录口径的细节差异。ICSV=0、RCSV≈1、AGCV≈0.33 均符合结构预期。

---

## 9. 输出文件清单（均在 `structure_audit_paired_20261105/`）

必需项：
- `partner_chain_manifest.csv` — 左右链映射、来源、partition 校验、原子行数
- `paired/pairwise_S1131_left.csv` / `..._right.csv`、`paired/pairwise_M1101_left.csv` / `..._right.csv` — 逐对双向 TM/coverage + `S_p/C_p`
- `paired/pairwise_{ds}_combined.csv` — `S_pair`,`C_pair`,`exposed_pair`
- `paired_complex_similarity_matrix.csv`（另附 `_coverage`、`_S_left`、`_S_right`）
- `per_fold_paired_exposure.csv`、`per_seed_paired_exposure.csv`、`per_eval_complex_paired.csv`
- `supplementary_table_S1_paired_exposure.csv`
- `failed_or_undetermined.csv`（空）

辅助与可复现：
- `partner_chain_conflicts.csv`（含 1KTZ/HM_1KTZ 角色交换记录）
- `paired/{ds}_{side}_self.csv`（自匹配 QC）
- `matrix_summary.json`、`params.json`、`software_provenance.json`
- `file_hashes.csv`（sha256）、`scripts/`（全部脚本）、`logs/`（`run_*.log`、`paired_exposure.log`、`qc_diagnostics.log`、`usalign_raw/` 每次 US-align 原始输出）、`README.md`

## 10. 复现命令

```bash
cd structure_audit_paired_20261105
python3 scripts/build_partners.py
for ds in S1131 M1101; do for side in left right; do
  python3 scripts/run_paired_side.py --dataset $ds --side $side --workers 64
done; done
python3 scripts/paired_exposure.py
python3 scripts/qc_diagnostics.py
```

完成。数据与脚本已提交，论文正文未改动。
