# 最终验收与质量确认报告（Paired global-partner structural exposure）

> 目录：`structure_audit_paired_20261105/`
> 状态：**最终验收**。不再扩展研究范围。未改动 `structure_audit_20261105/`、未动原始数据、未重生成划分、未重训模型。
> US-align 20260328，命令 `USalign <partner_i> <partner_j> -mm 1 -ter 1`；阈值 τ=0.70，c=0.80；HM 结构独立。
> 配套数据文件：`FINAL_TABLE_S1.csv`（本报告所有数字的机器可读版本，含分子/分母/各口径）。

---

## 1. 指标定义：分子、分母、聚合方式（逐项固定，不再有歧义）

对验证复合物 v、训练复合物 t、交互侧 p∈{left,right}：

- `S_p(v,t)=min(TM_{p,v},TM_{p,t})`，`C_p(v,t)=min(C_{p,v},C_{p,t})`
- `S_pair(v,t)=min(S_left,S_right)`，`C_pair(v,t)=min(C_left,C_right)`
- **E_struct(v)=1** 当且仅当存在**同一个** t 同时满足 `S_pair≥0.70` 且 `C_pair≥0.80`（左右两侧**不跨 t 混用**）。

**记录级暴露（record-weighted）**：一条评估记录“暴露” ⇔ 其所属复合物 E_struct=1。
- **分子** = Σ（E_struct=1 的验证复合物所含评估记录数）
- **分母** = **全部有效评估记录数**（resolved；本数据 UNRESOLVED=0，故 resolved=全部）
- 复合物级暴露（complex-weighted）：分子=暴露复合物数，分母=全部有效评估复合物数。

**聚合口径（本次统一）**：
- **主结果（MAIN，正文）**：**seed-pooled record-weighted**。每个 seed 把自己的 5 个 fold 合并（Σ暴露记录 / Σ全部记录）得到一个比值；再对 5 个 seed 取**均值与样本标准差（n=5，ddof=1）**。
- **补充统计（SUPP）**：①25 个 fold 的 fold 级记录加权值的**算术平均与样本标准差（n=25）**；②complex-weighted exposure。
- 报告同时给出 **grand pooled**（把全部 seed×fold 的分子/分母直接相除）作为交叉验证，其数值与 MAIN 在“每复合物每 seed 恰为一次评估”的数据结构下一致（下表逐行核对）。

**RCSV 双口径**：RCSV 允许相同 PDB 跨折（预期暴露），故同时给
- 含相同 PDB 匹配（**MAIN**）；
- 排除相同 PDB 匹配：从候选训练复合物中剔除该验证复合物自身，但**分母仍为全部有效评估记录**（不缩小分母）。

**PCSV/ICSV**：无相同 PDB 跨折，含/排除两种口径**构造上完全相等**（下表已核对：分子分母相同）。

---

## 2. 最终推荐数值表（正文主结果 = seed-pooled record-weighted，含相同复合物）

| 数据集 | 划分 | **推荐值（seed-pooled recW, 含相同）mean ± SD(n=5)** | 分子/分母（grand pooled） | 排除相同 recW mean ± SD(n=5) | 25-fold recW mean ± SD(n=25) | cpx 含相同 seed-pooled ± SD | cpx 含相同 25-fold ± SD |
|---|---|---|---|---|---|---|---|
| S1131 | RCSV | **0.992 ± 0.002** | 5611 / 5655 = 0.992 | 0.460 ± 0.001（分母 5655） | 0.992 ± 0.006 | 0.972 ± 0.005 | 0.972 ± 0.018 |
| S1131 | PCSV | **0.447 ± 0.015** | 2528 / 5655 = 0.447 | 0.447 ± 0.015（同 MAIN） | 0.447 ± 0.315 | 0.641 ± 0.016 | 0.644 ± 0.077 |
| S1131 | ICSV | **0.000 ± 0.000** | 0 / 5655 = 0.000 | 0.000（同 MAIN） | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| M1101 | RCSV | **1.000 ± 0.000** | 5505 / 5505 = 1.000 | 0.628 ± 0.000（分母 5505） | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| M1101 | PCSV | **0.589 ± 0.036** | 3244 / 5505 = 0.589 | 0.589 ± 0.036（同 MAIN） | 0.577 ± 0.254 | 0.525 ± 0.034 | 0.607 ± 0.224 |
| M1101 | ICSV | **0.000 ± 0.000** | 0 / 5505 = 0.000 | 0.000（同 MAIN） | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| M1101 | AGCV | **0.350 ± 0.000** | 1925 / 5505 = 0.350 | 0.350 ± 0.000（同 MAIN） | 0.330 ± 0.370 | 0.219 ± 0.000 | 0.358 ± 0.355 |

说明：S1131 采用 112 个复合物、5655 条评估记录（5 seed×5 fold 累积，每复合物每 seed 评估一次）；M1101 采用 32 个复合物、5505 条记录。RCSV 的 MAIN≈1 来自“相同 PDB 跨折”这一预期暴露（定义内，不删除）；其“排除相同”列才是真正的跨复合物结构暴露。

---

## 3. AGCV 的 0.330 与 0.350：分别对应什么口径（不得混用）

- **0.350 = seed-pooled record-weighted exposure（MAIN）**：每个 seed 合并 5 fold 后得 0.3497，5 个 seed 均为 0.3497 → 均值 0.350、SD=0。这是**正文推荐值**。
- **0.330 = 25 个 fold 的 fold 级记录加权的算术平均（SUPP）**：0.330 ± 0.370。SD 很大是因为 AGCV 每个 fold 仅 ~5–6 个评估复合物，fold 间记录数不均、单 fold 可能全暴露或全不暴露，导致 fold 级方差极大；把 25 个比值做**非加权平均**会被小样本 fold 拉低到 0.330。

两者差异源于**聚合方式不同**（先合并再平均 vs 直接对 fold 比值平均），**不是两种数据结构或两套阈值**。正文一律使用 **0.350（seed-pooled，n=5）**；0.330 只作为“fold 级变异性”的补充说明。同理，M1101 complex-weighted 的 0.219（seed-pooled）与 0.358（25-fold 平均）也是**同一量的两种聚合**，正文 complex-weighted 采用 seed-pooled/grand-pooled（0.219），fold 级平均仅作方差展示。

---

## 4. 1KTZ / HM_1KTZ 链角色冲突：最终敏感性核查

**主映射（recommended，依据原始生物学注释与 metadata）**：
- `1KTZ`：left(抗体侧)=**B**（TGF-β II 型受体），right(抗原侧)=**A**（TGF-β3 配体）。
- `HM_1KTZ`：left=**B**（受体），right=**A**（TGF-β1 配体）。
- 依据：`antigen_mapping.csv` 明确 `antibody_chains=B / antigen_chains=A`，`antigen_name=TGF-beta3`，`biological_role=non_antibody_complex`，`construct_information` 指出链 A 为配体、链 B 为受体。CSV 列 `Heavy=A / Antigen=B` 与之相反，属列名约定差异。

**替代映射（CSV 列，敏感性）**：把这两个复合物左右对调（left=A，right=B）。

**核查方式**：只对涉及 1KTZ/HM_1KTZ 的 62 对做**必要局部比对**（现成 partner PDB，未重跑全对全矩阵），其余对复用现有 combined 矩阵，重算 AGCV 暴露。结果见 `sensitivity_1KTZ_alt_pairwise.csv` 与 `logs/final_sensitivity_1ktz.log`。

**结论**：
- AGCV 主映射 seed-pooled recW = **0.3497**；替代映射 = **0.3497**；**Δ = +0.0000**，**0 个暴露判定翻转**。
- 原因：1KTZ 与 HM_1KTZ 唯一的结构近邻就是彼此；两者**同时**对调左右时，彼此比对仍是“A↔A、B↔B”，`S_pair=min(S_left,S_right)` 不变；对普通复合物在任一映射下均不达阈值。
- **主映射稳健，AGCV 结果不依赖该列名约定**；因此推荐采用 metadata 生物角色为主映射。

---

## 5. 数据、脚本与完整性（保留不变）

- 旧整复合物目录 `structure_audit_20261105/` **原样保留、未改动**。
- 本次新增/最终产物（均在 paired 目录）：`FINAL_TABLE_S1.csv`、本报告、`sensitivity_1KTZ_alt_pairwise.csv`、脚本 `final_tables.py`、`final_sensitivity_1ktz.py`，以及此前全部计算产物。
- 未修改原始数据、未重生成划分、未重训模型。
- 文件哈希：`file_hashes.csv`（sha256，覆盖除 `logs/usalign_raw/` 与自身外的全部文件）。
- 失败/未决：`failed_or_undetermined.csv` 为空（无 US-align 失败、无 UNRESOLVED）。

**已知限制（明确标注，不再扩展）**：
1. 记录的“复合物归属”以划分文件的 PDB 字段为准；若某记录 PDB 无法映射（本数据为 0），才计为 UNRESOLVED，且不计入 NOT_EXPOSED。
2. S1131 的复合物数（112）< 记录数（5655），记录/复合物加权数值差异已单列，正文主用 record-weighted。
3. fold 级方差（n=25）在小 fold 上偏大，故不作正文主值，仅作补充。

---

## 6. 可直接写入论文 Section 3.1 的英文段落

> **Paired global-partner structural exposure.** We quantified cross-split structural exposure at the level of each interaction partner rather than the whole complex. For every validation–training complex pair, the left and right partner chains were aligned separately with US-align (multi-chain mode), and directional TM-scores were normalized by the corresponding validation and training partner lengths, giving a symmetric per-partner similarity `S_p = min(TM_{p,v}, TM_{p,t})` and bidirectional coverage `C_p = min(C_{p,v}, C_{p,t})`. A complex pair was scored by the stricter of the two sides, `S_pair = min(S_left, S_right)` and `C_pair = min(C_left, C_right)`. A validation complex was considered structurally exposed only if a **single** training complex satisfied both `S_pair ≥ 0.70` and `C_pair ≥ 0.80` for both partners; left and right best matches were never drawn from different training complexes. We report seed-pooled record-weighted exposure (mean ± sample SD across five seeds), with identical-complex matches retained for RCSV as expected exposure and additionally reported when excluded. Under random cross-validation (RCSV) exposure was 99.2 ± 0.2% for S1131 and 100% for M1101 (excluding identical complexes, 46.0 ± 0.1% and 62.8%, respectively). PDB-level cross-validation (PCSV) removed identical-identifier overlap yet retained substantial cross-entry bilateral similarity: 44.7 ± 1.5% for S1131 and 58.9 ± 3.6% for M1101. Identical-partner-sequence clustering (ICSV) eliminated paired structural exposure entirely (0.0% in both datasets). Antigen-group-grouped cross-validation (AGCV) of M1101 still left measurable structural neighbours (35.0%). All exposure estimates were robust to a documented alternative left/right chain-role assignment for the 1KTZ/HM_1KTZ entries (Δ = 0.000).

---

## 7. Supplementary Table S1（最终数值，与 `FINAL_TABLE_S1.csv` 一致）

正文主结果（seed-pooled record-weighted exposure，含相同复合物；均值 ± 样本标准差 n=5；n 为 seed 数）：

| Model dataset | Split | Paired structural exposure (record-weighted, %) | 95% 取值参考（n=5 SD） | Excluding identical (%) |
|---|---|---|---|---|
| S1131 | RCSV | 99.2 | ± 0.2 | 46.0 |
| S1131 | PCSV | 44.7 | ± 1.5 | 44.7 |
| S1131 | ICSV | 0.0 | ± 0.0 | 0.0 |
| M1101 | RCSV | 100.0 | ± 0.0 | 62.8 |
| M1101 | PCSV | 58.9 | ± 3.6 | 58.9 |
| M1101 | ICSV | 0.0 | ± 0.0 | 0.0 |
| M1101 | AGCV | 35.0 | ± 0.0 | 35.0 |

补充：25-fold 记录加权算术平均（均值 ± SD, n=25）为 S1131 RCSV 99.2±0.6、PCSV 44.7±31.5、ICSV 0.0；M1101 RCSV 100.0、PCSV 57.7±25.4、ICSV 0.0、AGCV 33.0±37.0；complex-weighted（seed-pooled）对应 97.2、64.1、0.0、100.0、52.5、0.0、21.9。完整机器可读数据见 `FINAL_TABLE_S1.csv`。

---

任务结束。未提出新的检查或研究建议。
