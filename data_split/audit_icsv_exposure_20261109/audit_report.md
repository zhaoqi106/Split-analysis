# ICSV 跨折分子暴露审计报告

本报告由 `audit_crossfold_exposure.py` 生成，**只读**：不修改任何正式 ICSV 划分、`icsv_group_map.csv`、结构文件或模型产物；发现问题只做记录，未擅自修复。

## 任务 A：审计对象（ICSV 清单校验）

| 数据集 | master 行数 | 唯一ID | 唯一PDB | 唯一complex | 构造组数 |
|---|---|---|---|---|---|
| M1101 | 1101 | 1096 | 32 | 32 | 8 |
| S1131 | 1131 | 1131 | 112 | 112 | 36 |

全部文件级清单（路径/行数/MD5/唯一ID/PDB/组数）见 `icsv_manifest_files.csv`；每个 fold 的 train/val 数量见 `icsv_manifest_folds.csv`；划分完整性（val∪train=全集、五折 val 互斥覆盖全部记录）见 `audit_target_check.csv`。

## 任务 B：结论摘要

- 审计单元：50 个（2 数据集 × 5 seed × 5 折），validation 记录合计 11160。

| 分子层面 | 检查方法 | 跨折暴露记录 | 结论 |
|---|---|---|---|
| L1 拼接侧 partner 序列（官方构造单元） | MMseqs2 30%/80%双向 + 精确 | 0（精确 0） | 无暴露 |
| L2 单链 partner 序列 | MMseqs2 30%/80%双向 + 精确 | 638（精确 506，≥90% 506） | 存在暴露（重点） |
| L3 构造组 | group split val/train | 0 处组重叠 / 0 组跨 fold | 完整 |
| L4 记录身份 | 同 PDB / 同突变ID | PDB 0 / ID 0 | 无共用 |

> 关键发现：官方 ICSV 的构造单元是 **顺序敏感的拼接序列**（R1..R6 | L1..L3）。当两个复合物共享同一抗体/抗原单链、但链的列顺序不同或其它链不同时，拼接序列可能达不到 80% 双向覆盖而落入不同构造组，从而被分到不同折——表面看 L1 无暴露，但 **单链层面（L2）存在真实分子重复**。这正是仅检查文件分组会遗漏的跨折分子暴露。

### 分数据集汇总（对全部 seed×fold 求和）

| 数据集 | val记录合计 | 拼接侧暴露 | 单链暴露 | 单链精确 | 单链≥90% | 单链暴露率(记录) |
|---|---|---|---|---|---|---|
| M1101 | 5505 | 0 | 550 | 470 | 470 | 0.0999 |
| S1131 | 5655 | 0 | 88 | 36 | 36 | 0.0156 |

## 逐数据集×seed×fold 暴露明细

| 数据集 | seed | fold | val记录 | 拼接侧暴露 | 单链暴露 | 单链率 | 单链精确 | 单链≥90% | 组重叠 | 同PDB | 同ID | 结构近同对 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| M1101 | 0 | 0 | 683 | 0 | 71 | 0.1040 | 55 | 55 | 0 | 0 | 0 | 8 |
| M1101 | 0 | 1 | 271 | 0 | 0 | 0.0000 | 0 | 0 | 0 | 0 | 0 | 0 |
| M1101 | 0 | 2 | 51 | 0 | 0 | 0.0000 | 0 | 0 | 0 | 0 | 0 | 0 |
| M1101 | 0 | 3 | 48 | 0 | 39 | 0.8125 | 39 | 39 | 0 | 0 | 0 | 8 |
| M1101 | 0 | 4 | 48 | 0 | 0 | 0.0000 | 0 | 0 | 0 | 0 | 0 | 0 |
| M1101 | 42 | 0 | 683 | 0 | 71 | 0.1040 | 55 | 55 | 0 | 0 | 0 | 8 |
| M1101 | 42 | 1 | 271 | 0 | 0 | 0.0000 | 0 | 0 | 0 | 0 | 0 | 0 |
| M1101 | 42 | 2 | 51 | 0 | 0 | 0.0000 | 0 | 0 | 0 | 0 | 0 | 0 |
| M1101 | 42 | 3 | 48 | 0 | 39 | 0.8125 | 39 | 39 | 0 | 0 | 0 | 8 |
| M1101 | 42 | 4 | 48 | 0 | 0 | 0.0000 | 0 | 0 | 0 | 0 | 0 | 0 |
| M1101 | 1024 | 0 | 683 | 0 | 71 | 0.1040 | 55 | 55 | 0 | 0 | 0 | 8 |
| M1101 | 1024 | 1 | 271 | 0 | 0 | 0.0000 | 0 | 0 | 0 | 0 | 0 | 0 |
| M1101 | 1024 | 2 | 51 | 0 | 0 | 0.0000 | 0 | 0 | 0 | 0 | 0 | 0 |
| M1101 | 1024 | 3 | 48 | 0 | 39 | 0.8125 | 39 | 39 | 0 | 0 | 0 | 8 |
| M1101 | 1024 | 4 | 48 | 0 | 0 | 0.0000 | 0 | 0 | 0 | 0 | 0 | 0 |
| M1101 | 1234 | 0 | 683 | 0 | 71 | 0.1040 | 55 | 55 | 0 | 0 | 0 | 8 |
| M1101 | 1234 | 1 | 271 | 0 | 0 | 0.0000 | 0 | 0 | 0 | 0 | 0 | 0 |
| M1101 | 1234 | 2 | 51 | 0 | 0 | 0.0000 | 0 | 0 | 0 | 0 | 0 | 0 |
| M1101 | 1234 | 3 | 48 | 0 | 39 | 0.8125 | 39 | 39 | 0 | 0 | 0 | 8 |
| M1101 | 1234 | 4 | 48 | 0 | 0 | 0.0000 | 0 | 0 | 0 | 0 | 0 | 0 |
| M1101 | 2026 | 0 | 683 | 0 | 71 | 0.1040 | 55 | 55 | 0 | 0 | 0 | 8 |
| M1101 | 2026 | 1 | 271 | 0 | 0 | 0.0000 | 0 | 0 | 0 | 0 | 0 | 0 |
| M1101 | 2026 | 2 | 51 | 0 | 0 | 0.0000 | 0 | 0 | 0 | 0 | 0 | 0 |
| M1101 | 2026 | 3 | 48 | 0 | 39 | 0.8125 | 39 | 39 | 0 | 0 | 0 | 8 |
| M1101 | 2026 | 4 | 48 | 0 | 0 | 0.0000 | 0 | 0 | 0 | 0 | 0 | 0 |
| S1131 | 0 | 0 | 621 | 0 | 0 | 0.0000 | 0 | 0 | 0 | 0 | 0 | 0 |
| S1131 | 0 | 1 | 128 | 0 | 0 | 0.0000 | 0 | 0 | 0 | 0 | 0 | 0 |
| S1131 | 0 | 2 | 128 | 0 | 15 | 0.1172 | 5 | 5 | 0 | 0 | 0 | 0 |
| S1131 | 0 | 3 | 128 | 0 | 10 | 0.0781 | 7 | 7 | 0 | 0 | 0 | 0 |
| S1131 | 0 | 4 | 126 | 0 | 0 | 0.0000 | 0 | 0 | 0 | 0 | 0 | 0 |
| S1131 | 42 | 0 | 621 | 0 | 0 | 0.0000 | 0 | 0 | 0 | 0 | 0 | 0 |
| S1131 | 42 | 1 | 128 | 0 | 0 | 0.0000 | 0 | 0 | 0 | 0 | 0 | 0 |
| S1131 | 42 | 2 | 128 | 0 | 0 | 0.0000 | 0 | 0 | 0 | 0 | 0 | 0 |
| S1131 | 42 | 3 | 128 | 0 | 3 | 0.0234 | 0 | 0 | 0 | 0 | 0 | 0 |
| S1131 | 42 | 4 | 126 | 0 | 10 | 0.0794 | 0 | 0 | 0 | 0 | 0 | 0 |
| S1131 | 1024 | 0 | 621 | 0 | 0 | 0.0000 | 0 | 0 | 0 | 0 | 0 | 0 |
| S1131 | 1024 | 1 | 128 | 0 | 0 | 0.0000 | 0 | 0 | 0 | 0 | 0 | 0 |
| S1131 | 1024 | 2 | 128 | 0 | 5 | 0.0391 | 5 | 5 | 0 | 0 | 0 | 0 |
| S1131 | 1024 | 3 | 128 | 0 | 10 | 0.0781 | 7 | 7 | 0 | 0 | 0 | 0 |
| S1131 | 1024 | 4 | 126 | 0 | 10 | 0.0794 | 0 | 0 | 0 | 0 | 0 | 0 |
| S1131 | 1234 | 0 | 621 | 0 | 0 | 0.0000 | 0 | 0 | 0 | 0 | 0 | 0 |
| S1131 | 1234 | 1 | 128 | 0 | 0 | 0.0000 | 0 | 0 | 0 | 0 | 0 | 0 |
| S1131 | 1234 | 2 | 128 | 0 | 15 | 0.1172 | 5 | 5 | 0 | 0 | 0 | 0 |
| S1131 | 1234 | 3 | 128 | 0 | 10 | 0.0781 | 7 | 7 | 0 | 0 | 0 | 0 |
| S1131 | 1234 | 4 | 126 | 0 | 0 | 0.0000 | 0 | 0 | 0 | 0 | 0 | 0 |
| S1131 | 2026 | 0 | 621 | 0 | 0 | 0.0000 | 0 | 0 | 0 | 0 | 0 | 0 |
| S1131 | 2026 | 1 | 128 | 0 | 0 | 0.0000 | 0 | 0 | 0 | 0 | 0 | 0 |
| S1131 | 2026 | 2 | 128 | 0 | 0 | 0.0000 | 0 | 0 | 0 | 0 | 0 | 0 |
| S1131 | 2026 | 3 | 128 | 0 | 0 | 0.0000 | 0 | 0 | 0 | 0 | 0 | 0 |
| S1131 | 2026 | 4 | 126 | 0 | 0 | 0.0000 | 0 | 0 | 0 | 0 | 0 | 0 |

全部匹配记录见 `crossfold_chain_hits.csv`（单链）与 `crossfold_sequence_hits.csv`（拼接侧）；逐记录暴露见 `record_exposure.csv`；逐数据集×seed 暴露率汇总见 `exposure_by_dataset_seed.csv`；精确重叠见 `chain_exact_overlap.csv` / `concat_exact_overlap.csv`。

## 构造组完整性（L3）

- 每个构造组都完整落在单一 held-out 折；任何 fold 内 val/train 的组交集为 0。构造组层面完整。

## 重点核查：S1131 的 1E96 与 2A9K

| 数据集 | seed | PDB | 所属构造组 | 与对方同组 | held-out折 | 两目标始终同侧 | 链级跨折相似伙伴(节选) |
|---|---|---|---|---|---|---|---|
| S1131 | 0 | 1E96 | 1E96__A_B | True | 3 | True | seed0f2:TRAIN~31.6%~1HE8;seed0f2:TRAIN~31.6%~1LFD;seed0f3:VAL~31.6%~1HE8;seed0f3:VAL~31.6%~1LFD;seed1024f3:VAL~31.6%~1HE8;seed1024f3:VAL~31.6%~1LFD;seed1024f4:T |
| S1131 | 0 | 2A9K | 1E96__A_B | True | 3 | True | seed0f2:TRAIN~54.3%~1HE8;seed0f2:TRAIN~54.5%~1LFD;seed0f3:VAL~54.3%~1HE8;seed0f3:VAL~54.5%~1LFD;seed1024f3:VAL~54.3%~1HE8;seed1024f3:VAL~54.5%~1LFD;seed1024f4:T |
| S1131 | 42 | 1E96 | 1E96__A_B | True | 3 | True | seed0f2:TRAIN~31.6%~1HE8;seed0f2:TRAIN~31.6%~1LFD;seed0f3:VAL~31.6%~1HE8;seed0f3:VAL~31.6%~1LFD;seed1024f3:VAL~31.6%~1HE8;seed1024f3:VAL~31.6%~1LFD;seed1024f4:T |
| S1131 | 42 | 2A9K | 1E96__A_B | True | 3 | True | seed0f2:TRAIN~54.3%~1HE8;seed0f2:TRAIN~54.5%~1LFD;seed0f3:VAL~54.3%~1HE8;seed0f3:VAL~54.5%~1LFD;seed1024f3:VAL~54.3%~1HE8;seed1024f3:VAL~54.5%~1LFD;seed1024f4:T |
| S1131 | 1024 | 1E96 | 1E96__A_B | True | 3 | True | seed0f2:TRAIN~31.6%~1HE8;seed0f2:TRAIN~31.6%~1LFD;seed0f3:VAL~31.6%~1HE8;seed0f3:VAL~31.6%~1LFD;seed1024f3:VAL~31.6%~1HE8;seed1024f3:VAL~31.6%~1LFD;seed1024f4:T |
| S1131 | 1024 | 2A9K | 1E96__A_B | True | 3 | True | seed0f2:TRAIN~54.3%~1HE8;seed0f2:TRAIN~54.5%~1LFD;seed0f3:VAL~54.3%~1HE8;seed0f3:VAL~54.5%~1LFD;seed1024f3:VAL~54.3%~1HE8;seed1024f3:VAL~54.5%~1LFD;seed1024f4:T |
| S1131 | 1234 | 1E96 | 1E96__A_B | True | 3 | True | seed0f2:TRAIN~31.6%~1HE8;seed0f2:TRAIN~31.6%~1LFD;seed0f3:VAL~31.6%~1HE8;seed0f3:VAL~31.6%~1LFD;seed1024f3:VAL~31.6%~1HE8;seed1024f3:VAL~31.6%~1LFD;seed1024f4:T |
| S1131 | 1234 | 2A9K | 1E96__A_B | True | 3 | True | seed0f2:TRAIN~54.3%~1HE8;seed0f2:TRAIN~54.5%~1LFD;seed0f3:VAL~54.3%~1HE8;seed0f3:VAL~54.5%~1LFD;seed1024f3:VAL~54.3%~1HE8;seed1024f3:VAL~54.5%~1LFD;seed1024f4:T |
| S1131 | 2026 | 1E96 | 1E96__A_B | True | 4 | True | seed0f2:TRAIN~31.6%~1HE8;seed0f2:TRAIN~31.6%~1LFD;seed0f3:VAL~31.6%~1HE8;seed0f3:VAL~31.6%~1LFD;seed1024f3:VAL~31.6%~1HE8;seed1024f3:VAL~31.6%~1LFD;seed1024f4:T |
| S1131 | 2026 | 2A9K | 1E96__A_B | True | 4 | True | seed0f2:TRAIN~54.3%~1HE8;seed0f2:TRAIN~54.5%~1LFD;seed0f3:VAL~54.3%~1HE8;seed0f3:VAL~54.5%~1LFD;seed1024f3:VAL~54.3%~1HE8;seed1024f3:VAL~54.5%~1LFD;seed1024f4:T |

> 1E96 与 2A9K 同属一个构造组（`1E96__A_B`），彼此不会被拆到两侧。但链级审计显示该组与 S1131 其它 PDB（如 1LFD、1HE8）在 30–55% 同一性、≈95% 覆盖下相似，而后者落在别的折——即该蛋白家族存在跨折暴露。详见 `special_1E96_2A9K.csv`。

## 结构相似性（L5，US-align TM-score）

- 对每个数据集全部唯一 PDB 的 `wt` 结构做了 all-vs-all US-align，共 6856 条 PDB×PDB 记录。
- 分层：TM≥0.9 近同结构；≥0.7 相似；≥0.5 同 fold。
- 完整 TM 矩阵 `structural_pdb_tm_matrix.csv`；跨折结构对明细 `structural_crossfold_pairs.csv`。
- 说明：抗体类结构共享 Ig 折叠，TM≥0.5 可能只是同 fold 而非重复；因此真正的结构重复应结合 `structural_crossfold_pairs.csv` 中的 near_identical 层与链级审计交叉判断。

## 异常汇总

| 数据集 | seed | fold | 类型 | 严重度 | 明细 |
|---|---|---|---|---|---|
| M1101 | 0 | 0 | CHAIN_EXACT_OVERLAP | CRITICAL | 55/683 val records share an exact partner CHAIN with training (rate 0.1040) |
| M1101 | 0 | 0 | CHAIN_SIMILARITY(30-90%) | WARN | 16 val records have a chain 30-90% identical (>=30%/80% bicov) to training |
| M1101 | 0 | 0 | STRUCT_NEAR_IDENTICAL_CROSSFOLD | HIGH | 8 cross-fold PDB pairs with TM>=0.9 |
| M1101 | 0 | 0 | STRUCT_SIMILAR_CROSSFOLD | INFO | 3 cross-fold PDB pairs with TM in [0.7,0.9) |
| M1101 | 0 | 0 | STRUCT_SAME_FOLD_CROSSFOLD | INFO | 13 cross-fold PDB pairs with TM in [0.5,0.7) |
| M1101 | 0 | 1 | STRUCT_SAME_FOLD_CROSSFOLD | INFO | 14 cross-fold PDB pairs with TM in [0.5,0.7) |
| M1101 | 0 | 3 | CHAIN_EXACT_OVERLAP | CRITICAL | 39/48 val records share an exact partner CHAIN with training (rate 0.8125) |
| M1101 | 0 | 3 | STRUCT_NEAR_IDENTICAL_CROSSFOLD | HIGH | 8 cross-fold PDB pairs with TM>=0.9 |
| M1101 | 0 | 3 | STRUCT_SIMILAR_CROSSFOLD | INFO | 3 cross-fold PDB pairs with TM in [0.7,0.9) |
| M1101 | 0 | 3 | STRUCT_SAME_FOLD_CROSSFOLD | INFO | 3 cross-fold PDB pairs with TM in [0.5,0.7) |
| M1101 | 42 | 0 | CHAIN_EXACT_OVERLAP | CRITICAL | 55/683 val records share an exact partner CHAIN with training (rate 0.1040) |
| M1101 | 42 | 0 | CHAIN_SIMILARITY(30-90%) | WARN | 16 val records have a chain 30-90% identical (>=30%/80% bicov) to training |
| M1101 | 42 | 0 | STRUCT_NEAR_IDENTICAL_CROSSFOLD | HIGH | 8 cross-fold PDB pairs with TM>=0.9 |
| M1101 | 42 | 0 | STRUCT_SIMILAR_CROSSFOLD | INFO | 3 cross-fold PDB pairs with TM in [0.7,0.9) |
| M1101 | 42 | 0 | STRUCT_SAME_FOLD_CROSSFOLD | INFO | 13 cross-fold PDB pairs with TM in [0.5,0.7) |
| M1101 | 42 | 1 | STRUCT_SAME_FOLD_CROSSFOLD | INFO | 14 cross-fold PDB pairs with TM in [0.5,0.7) |
| M1101 | 42 | 3 | CHAIN_EXACT_OVERLAP | CRITICAL | 39/48 val records share an exact partner CHAIN with training (rate 0.8125) |
| M1101 | 42 | 3 | STRUCT_NEAR_IDENTICAL_CROSSFOLD | HIGH | 8 cross-fold PDB pairs with TM>=0.9 |
| M1101 | 42 | 3 | STRUCT_SIMILAR_CROSSFOLD | INFO | 3 cross-fold PDB pairs with TM in [0.7,0.9) |
| M1101 | 42 | 3 | STRUCT_SAME_FOLD_CROSSFOLD | INFO | 3 cross-fold PDB pairs with TM in [0.5,0.7) |
| M1101 | 1024 | 0 | CHAIN_EXACT_OVERLAP | CRITICAL | 55/683 val records share an exact partner CHAIN with training (rate 0.1040) |
| M1101 | 1024 | 0 | CHAIN_SIMILARITY(30-90%) | WARN | 16 val records have a chain 30-90% identical (>=30%/80% bicov) to training |
| M1101 | 1024 | 0 | STRUCT_NEAR_IDENTICAL_CROSSFOLD | HIGH | 8 cross-fold PDB pairs with TM>=0.9 |
| M1101 | 1024 | 0 | STRUCT_SIMILAR_CROSSFOLD | INFO | 3 cross-fold PDB pairs with TM in [0.7,0.9) |
| M1101 | 1024 | 0 | STRUCT_SAME_FOLD_CROSSFOLD | INFO | 13 cross-fold PDB pairs with TM in [0.5,0.7) |
| M1101 | 1024 | 1 | STRUCT_SAME_FOLD_CROSSFOLD | INFO | 14 cross-fold PDB pairs with TM in [0.5,0.7) |
| M1101 | 1024 | 3 | CHAIN_EXACT_OVERLAP | CRITICAL | 39/48 val records share an exact partner CHAIN with training (rate 0.8125) |
| M1101 | 1024 | 3 | STRUCT_NEAR_IDENTICAL_CROSSFOLD | HIGH | 8 cross-fold PDB pairs with TM>=0.9 |
| M1101 | 1024 | 3 | STRUCT_SIMILAR_CROSSFOLD | INFO | 3 cross-fold PDB pairs with TM in [0.7,0.9) |
| M1101 | 1024 | 3 | STRUCT_SAME_FOLD_CROSSFOLD | INFO | 3 cross-fold PDB pairs with TM in [0.5,0.7) |
| M1101 | 1234 | 0 | CHAIN_EXACT_OVERLAP | CRITICAL | 55/683 val records share an exact partner CHAIN with training (rate 0.1040) |
| M1101 | 1234 | 0 | CHAIN_SIMILARITY(30-90%) | WARN | 16 val records have a chain 30-90% identical (>=30%/80% bicov) to training |
| M1101 | 1234 | 0 | STRUCT_NEAR_IDENTICAL_CROSSFOLD | HIGH | 8 cross-fold PDB pairs with TM>=0.9 |
| M1101 | 1234 | 0 | STRUCT_SIMILAR_CROSSFOLD | INFO | 3 cross-fold PDB pairs with TM in [0.7,0.9) |
| M1101 | 1234 | 0 | STRUCT_SAME_FOLD_CROSSFOLD | INFO | 13 cross-fold PDB pairs with TM in [0.5,0.7) |
| M1101 | 1234 | 1 | STRUCT_SAME_FOLD_CROSSFOLD | INFO | 14 cross-fold PDB pairs with TM in [0.5,0.7) |
| M1101 | 1234 | 3 | CHAIN_EXACT_OVERLAP | CRITICAL | 39/48 val records share an exact partner CHAIN with training (rate 0.8125) |
| M1101 | 1234 | 3 | STRUCT_NEAR_IDENTICAL_CROSSFOLD | HIGH | 8 cross-fold PDB pairs with TM>=0.9 |
| M1101 | 1234 | 3 | STRUCT_SIMILAR_CROSSFOLD | INFO | 3 cross-fold PDB pairs with TM in [0.7,0.9) |
| M1101 | 1234 | 3 | STRUCT_SAME_FOLD_CROSSFOLD | INFO | 3 cross-fold PDB pairs with TM in [0.5,0.7) |
| M1101 | 2026 | 0 | CHAIN_EXACT_OVERLAP | CRITICAL | 55/683 val records share an exact partner CHAIN with training (rate 0.1040) |
| M1101 | 2026 | 0 | CHAIN_SIMILARITY(30-90%) | WARN | 16 val records have a chain 30-90% identical (>=30%/80% bicov) to training |
| M1101 | 2026 | 0 | STRUCT_NEAR_IDENTICAL_CROSSFOLD | HIGH | 8 cross-fold PDB pairs with TM>=0.9 |
| M1101 | 2026 | 0 | STRUCT_SIMILAR_CROSSFOLD | INFO | 3 cross-fold PDB pairs with TM in [0.7,0.9) |
| M1101 | 2026 | 0 | STRUCT_SAME_FOLD_CROSSFOLD | INFO | 13 cross-fold PDB pairs with TM in [0.5,0.7) |
| M1101 | 2026 | 1 | STRUCT_SAME_FOLD_CROSSFOLD | INFO | 14 cross-fold PDB pairs with TM in [0.5,0.7) |
| M1101 | 2026 | 3 | CHAIN_EXACT_OVERLAP | CRITICAL | 39/48 val records share an exact partner CHAIN with training (rate 0.8125) |
| M1101 | 2026 | 3 | STRUCT_NEAR_IDENTICAL_CROSSFOLD | HIGH | 8 cross-fold PDB pairs with TM>=0.9 |
| M1101 | 2026 | 3 | STRUCT_SIMILAR_CROSSFOLD | INFO | 3 cross-fold PDB pairs with TM in [0.7,0.9) |
| M1101 | 2026 | 3 | STRUCT_SAME_FOLD_CROSSFOLD | INFO | 3 cross-fold PDB pairs with TM in [0.5,0.7) |
| S1131 | 0 | 0 | STRUCT_SAME_FOLD_CROSSFOLD | INFO | 51 cross-fold PDB pairs with TM in [0.5,0.7) |
| S1131 | 0 | 1 | STRUCT_SAME_FOLD_CROSSFOLD | INFO | 44 cross-fold PDB pairs with TM in [0.5,0.7) |
| S1131 | 0 | 2 | CHAIN_EXACT_OVERLAP | CRITICAL | 5/128 val records share an exact partner CHAIN with training (rate 0.1172) |
| S1131 | 0 | 2 | CHAIN_SIMILARITY(30-90%) | WARN | 10 val records have a chain 30-90% identical (>=30%/80% bicov) to training |
| S1131 | 0 | 2 | STRUCT_SAME_FOLD_CROSSFOLD | INFO | 6 cross-fold PDB pairs with TM in [0.5,0.7) |
| S1131 | 0 | 3 | CHAIN_EXACT_OVERLAP | CRITICAL | 7/128 val records share an exact partner CHAIN with training (rate 0.0781) |
| S1131 | 0 | 3 | CHAIN_SIMILARITY(30-90%) | WARN | 3 val records have a chain 30-90% identical (>=30%/80% bicov) to training |
| S1131 | 0 | 3 | STRUCT_SIMILAR_CROSSFOLD | INFO | 2 cross-fold PDB pairs with TM in [0.7,0.9) |
| S1131 | 0 | 3 | STRUCT_SAME_FOLD_CROSSFOLD | INFO | 27 cross-fold PDB pairs with TM in [0.5,0.7) |
| S1131 | 0 | 4 | STRUCT_SIMILAR_CROSSFOLD | INFO | 2 cross-fold PDB pairs with TM in [0.7,0.9) |
| S1131 | 0 | 4 | STRUCT_SAME_FOLD_CROSSFOLD | INFO | 6 cross-fold PDB pairs with TM in [0.5,0.7) |
| S1131 | 42 | 0 | STRUCT_SAME_FOLD_CROSSFOLD | INFO | 51 cross-fold PDB pairs with TM in [0.5,0.7) |
| S1131 | 42 | 1 | STRUCT_SAME_FOLD_CROSSFOLD | INFO | 45 cross-fold PDB pairs with TM in [0.5,0.7) |
| S1131 | 42 | 2 | STRUCT_SAME_FOLD_CROSSFOLD | INFO | 12 cross-fold PDB pairs with TM in [0.5,0.7) |
| S1131 | 42 | 3 | CHAIN_SIMILARITY(30-90%) | WARN | 3 val records have a chain 30-90% identical (>=30%/80% bicov) to training |
| S1131 | 42 | 3 | STRUCT_SAME_FOLD_CROSSFOLD | INFO | 28 cross-fold PDB pairs with TM in [0.5,0.7) |
| S1131 | 42 | 4 | CHAIN_SIMILARITY(30-90%) | WARN | 10 val records have a chain 30-90% identical (>=30%/80% bicov) to training |
| S1131 | 42 | 4 | STRUCT_SAME_FOLD_CROSSFOLD | INFO | 2 cross-fold PDB pairs with TM in [0.5,0.7) |
| S1131 | 1024 | 0 | STRUCT_SAME_FOLD_CROSSFOLD | INFO | 51 cross-fold PDB pairs with TM in [0.5,0.7) |
| S1131 | 1024 | 1 | STRUCT_SAME_FOLD_CROSSFOLD | INFO | 42 cross-fold PDB pairs with TM in [0.5,0.7) |
| S1131 | 1024 | 2 | CHAIN_EXACT_OVERLAP | CRITICAL | 5/128 val records share an exact partner CHAIN with training (rate 0.0391) |
| S1131 | 1024 | 2 | STRUCT_SIMILAR_CROSSFOLD | INFO | 2 cross-fold PDB pairs with TM in [0.7,0.9) |
| S1131 | 1024 | 2 | STRUCT_SAME_FOLD_CROSSFOLD | INFO | 9 cross-fold PDB pairs with TM in [0.5,0.7) |
| S1131 | 1024 | 3 | CHAIN_EXACT_OVERLAP | CRITICAL | 7/128 val records share an exact partner CHAIN with training (rate 0.0781) |
| S1131 | 1024 | 3 | CHAIN_SIMILARITY(30-90%) | WARN | 3 val records have a chain 30-90% identical (>=30%/80% bicov) to training |
| S1131 | 1024 | 3 | STRUCT_SAME_FOLD_CROSSFOLD | INFO | 26 cross-fold PDB pairs with TM in [0.5,0.7) |
| S1131 | 1024 | 4 | CHAIN_SIMILARITY(30-90%) | WARN | 10 val records have a chain 30-90% identical (>=30%/80% bicov) to training |
| S1131 | 1024 | 4 | STRUCT_SIMILAR_CROSSFOLD | INFO | 2 cross-fold PDB pairs with TM in [0.7,0.9) |
| S1131 | 1024 | 4 | STRUCT_SAME_FOLD_CROSSFOLD | INFO | 4 cross-fold PDB pairs with TM in [0.5,0.7) |
| S1131 | 1234 | 0 | STRUCT_SAME_FOLD_CROSSFOLD | INFO | 51 cross-fold PDB pairs with TM in [0.5,0.7) |
| S1131 | 1234 | 1 | STRUCT_SIMILAR_CROSSFOLD | INFO | 2 cross-fold PDB pairs with TM in [0.7,0.9) |
| S1131 | 1234 | 1 | STRUCT_SAME_FOLD_CROSSFOLD | INFO | 42 cross-fold PDB pairs with TM in [0.5,0.7) |
| S1131 | 1234 | 2 | CHAIN_EXACT_OVERLAP | CRITICAL | 5/128 val records share an exact partner CHAIN with training (rate 0.1172) |
| S1131 | 1234 | 2 | CHAIN_SIMILARITY(30-90%) | WARN | 10 val records have a chain 30-90% identical (>=30%/80% bicov) to training |
| S1131 | 1234 | 2 | STRUCT_SAME_FOLD_CROSSFOLD | INFO | 7 cross-fold PDB pairs with TM in [0.5,0.7) |
| S1131 | 1234 | 3 | CHAIN_EXACT_OVERLAP | CRITICAL | 7/128 val records share an exact partner CHAIN with training (rate 0.0781) |
| S1131 | 1234 | 3 | CHAIN_SIMILARITY(30-90%) | WARN | 3 val records have a chain 30-90% identical (>=30%/80% bicov) to training |
| S1131 | 1234 | 3 | STRUCT_SAME_FOLD_CROSSFOLD | INFO | 26 cross-fold PDB pairs with TM in [0.5,0.7) |
| S1131 | 1234 | 4 | STRUCT_SIMILAR_CROSSFOLD | INFO | 2 cross-fold PDB pairs with TM in [0.7,0.9) |
| S1131 | 1234 | 4 | STRUCT_SAME_FOLD_CROSSFOLD | INFO | 4 cross-fold PDB pairs with TM in [0.5,0.7) |
| S1131 | 2026 | 0 | STRUCT_SAME_FOLD_CROSSFOLD | INFO | 51 cross-fold PDB pairs with TM in [0.5,0.7) |
| S1131 | 2026 | 1 | STRUCT_SAME_FOLD_CROSSFOLD | INFO | 44 cross-fold PDB pairs with TM in [0.5,0.7) |
| S1131 | 2026 | 2 | STRUCT_SAME_FOLD_CROSSFOLD | INFO | 11 cross-fold PDB pairs with TM in [0.5,0.7) |
| S1131 | 2026 | 3 | STRUCT_SAME_FOLD_CROSSFOLD | INFO | 3 cross-fold PDB pairs with TM in [0.5,0.7) |
| S1131 | 2026 | 4 | STRUCT_SAME_FOLD_CROSSFOLD | INFO | 25 cross-fold PDB pairs with TM in [0.5,0.7) |

## 复现

```bash
cd SPLIT_AUDIT_ICSV_20261109
python3 audit_crossfold_exposure.py --data-root .. --out . --threads 64
```

阈值：MMseqs2 `--min-seq-id 0.30 -c 0.80 --cov-mode 0 -s 7.5`；US-align 默认参数取两向 TM 的较大值。运行环境见 `environment.txt`，完整命令与时间戳见 `run.log`。
