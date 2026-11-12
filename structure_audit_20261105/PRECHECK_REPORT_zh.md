# 结构审计预检报告（第一阶段，只读）

生成目录：`/data/structure_audit_20261105/`
数据集：S1131、M1101
软件：US-align 20260328（`~/.local/bin/USalign`，静态 x86-64）
说明：本报告只做只读预检与映射核验；正式比对在预检全部通过后执行，结果见 `STRUCTURE_AUDIT_RESULTS_zh.md`。

---

## 0. 预检结论（一句话）

预检**全部通过**：结构文件齐全、链映射与 CSV 完全一致、划分文件完整且训练/评估无重叠、HM 结构按独立条目保留、US-align 安装成功并已用多链测试验证。可以进入正式计算，且正式计算已完成、0 个比对失败。

---

## 1. WT / MUT 结构文件实际数量

| 数据集 | WT 全部文件 | WT 完整复合物 | WT 单链切片 | WT `.xvg` | MUT `.pdb` | MUT `.xvg` |
|---|---|---|---|---|---|---|
| S1131 | 112 | 112 | 0 | 0 | 1135 | 6 |
| M1101 | 88 | 32 | 24 | 32 | 1336 | 320 |

- S1131 的 WT 目录 112 个文件全部是完整复合物 `.pdb`，没有单链切片、没有 `.xvg`。
- M1101 的 88 个 WT 文件 = 32 个完整复合物 `.pdb` + 24 个单链切片（如 `1JRH.H.pdb`）+ 32 个 `.xvg`（界面/特征辅助文件，非结构本体，正式比对不使用）。
- MUT 仅用于映射核验：每个记录对应的 `ID + ".pdb"` 均存在（覆盖率 2232/2232 = 100%）。MUT 中出现额外单链切片（M1101 约 235 个）与 `.xvg`，**未混入 WT 审计**。

## 2. 每个数据集实际对应的独特 WT 复合物数量

- S1131：**112** 个独特 WT 复合物（112 个 PDB），对 1131 条记录。
- M1101：**32** 个独特 WT 复合物 = 27 个常规 PDB + 5 个 HM 结构，对 1101 条记录。

每个记录都能映射到唯一的 WT 复合物；`WT 完整复合物` 与 `CSV 引用 PDB` 双向完全一致：
缺失（被引用但没有结构）= 0，冗余（有结构但无记录）= 0。

## 3. 结构文件与 CSV 记录的映射

逐记录映射清单：`manifests/chain_mapping_audit.csv`（2232 行）。
逐复合物汇总：`manifests/structure_manifest.csv`（144 行 = 112 + 32）。

关键校验（全部 2232 条记录）：

| 校验项 | 结果 |
|---|---|
| 记录 → WT 完整复合物文件存在 | 2232 / 2232 |
| `Partners` 字段与 Heavy/Light/Antigen 链列一致 | 2232 / 2232 |
| 伙伴链全部出现在 WT 复合物文件里 | 2232 / 2232（缺失链 0） |
| 记录 → MUT 文件（`ID.pdb`）存在 | 2232 / 2232 |
| 复合物全局映射状态 | 2232 `OK` |

- WT 完整复合物里恰好只包含该复合物伙伴链，**没有额外多餘链**（144 个复合物全部满足），因此整复合物对齐即等价于“双侧伙伴全部参与”。
- 重复 mutation 记录（同一复合物的多个突变）正确共享同一个 WT 复合物结构。

## 4. 多链伙伴提取是否完整

按 CSV 的 `Partners` 与链列定义：
- 左侧伙伴 = `Heavy_chain + Light_chain`；右侧伙伴 = `Antigen_chain1 + Antigen_chain2`。
- S1131：所有复合物两侧均为**单链**伙伴（如 `A_B`、`E_I`），不存在多链伙伴。
- M1101：**26** 个复合物左侧为多链（抗体 H+L 等，如 `HL_A`、`AB_CD`、`BC_A`），**3** 个复合物右侧为多链（`1BJ1`、`1CZ8` 为 `HL_VW`；`1DVF` 为 `AB_CD`）。
- 所有伙伴链都能从 WT 完整复合物文件直接取到；8 个复合物另有预切单链文件（`1JRH`、`1N8Z`、`2JEL`、`2NY7`、`3BDY`、`3BE1`、`3BN9`、`HM_3BN9`），但全复合物对齐不依赖这些切片。

结论：多链伙伴提取完整、可验证。

## 5. HM 结构是否存在及其映射

存在 5 个 HM 结构，均作为**独立复合物**保留，未去掉 `HM_` 前缀、未与基础 PDB 合并：

| HM 复合物 | 记录数 | 基础 PDB | 基础 PDB 是否存在 |
|---|---|---|---|
| HM_1KTZ | 22 | 1KTZ | 是 |
| HM_1YY9 | 16 | 1YY9 | 是 |
| HM_2NYY | 25 | 2NYY | 是 |
| HM_2NZ9 | 16 | 2NZ9 | 是 |
| HM_3BN9 | 8 | 3BN9 | 是 |

HM 与基础 PDB 的逐对比较记录见 `exposure/hm_vs_base_all.csv` 与 `exposure/hm_template_summary.csv`（详见结果报告）。

## 6. 五个随机种子下各划分 CSV 是否完整

- 两个数据集 × 5 个种子 × {RCSV, PCSV, ICSV} 均齐全：每个目录 10 个文件（`train_fold_0..4`、`val_fold_0..4`）。
- M1101 每个种子另有 AGCV 目录，含 `fold_{k}_{outer_train,outer_eval,inner_train,inner_val}.csv` 共 20 个折文件。
- 逐折完整性（`precheck/csv_integrity_report.csv`，共 175 个 fold×seed 单元）：**全部 PASS**。
  - 每折 `train + val` 的多重集 = 该数据集完整记录（无丢失、无越界重叠）；
  - 5 折 `val` 并集 = 完整记录（每个记录恰好作为评估一次）。
- 外层划分独特性（按“记录→折号”完整映射判定）：

| 数据集 | 划分 | 5 个种子的独特外层划分 |
|---|---|---|
| S1131 | RCSV / PCSV / ICSV | 5 / 5 / 5 |
| M1101 | RCSV / PCSV / ICSV | 5 / 5 / **1** |
| M1101 | AGCV | **4**（种子 42 与 1234 相同） |

与既有正式分析记录一致：M1101 ICSV 五个种子是同一套外层划分；M1101 AGCV 只有 4 套独特外层划分，种子 42 与 1234 重复，统计时不得当作独立划分重复计数。

## 7. AGCV 的 outer/inner 对应

对 5 个种子 × 5 个外层折逐折核验（`csv_integrity_report.csv` 中 `mode=agcv`）：

- `outer_train` = `inner_train ∪ inner_val`：**全部成立**；
- `inner_train ∩ inner_val = ∅`：全部成立；
- `outer_train ∩ outer_eval = ∅`：全部成立；
- `outer_train ∪ outer_eval` = 全部 1101 条：全部成立；
- `inner_val / outer_train` 比例约 0.20（实际 0.201–0.210），符合“内层验证≈20%”设置。

未把 `inner_train` 误当作 `outer_train`；结构暴露只用 `outer_train` 与 `outer_eval` 计算，`inner_*` 仅用于内层完整性检查。

## 8. US-align 是否已经可用

- 服务器原 PATH / conda 环境 / HPC module 均**未预装** US-align（`command -v` 全部未命中，`/home`、`/opt`、`/usr/local`、`/share` 均无匹配）。
- 已按第二优先级安装官方 Linux 预编译二进制到用户目录：
  - 路径：`/data/tools/USalign`（另存副本 `usalign_bin/USalign`）；
  - 版本：`US-align (Version 20260328)`；
  - 架构：`ELF 64-bit LSB executable, x86-64, statically linked`，与服务器 `x86_64` 匹配；
  - SHA256：`dc6624938b7800a4bded5a7bde837b84ca04e4c14c7006ee0bf8a8fb83236408`；
  - 已 `chmod u+x`，`-v`/`-h` 正常。
- 多链比对模式已用真实结构验证（`-mm 1 -ter 1`）：
  - 自身对齐 TM=1.000；
  - 4 链抗体-抗原复合物 `1BJ1` vs `1CZ8`：TM 0.995，对齐 619/619；
  - 链数不同 `1DVF(4链)` vs `1T83(3链)`：程序能自动做链映射；
  - HM 与基础 PDB、同源抗体复合物均得到合理 TM。

## 9. 无管理员权限下能否完成安装

可以。已通过“官方 Linux 预编译二进制 → `$HOME/.local/bin` → `chmod u+x`”完成，全程未使用 sudo、未改系统目录、未改他人环境。服务器 CPU 为 x86_64，二进制静态链接，无需额外依赖。

## 10. 正式计算预计比对次数与资源

- 独特 WT 复合物：S1131 = 112，M1101 = 32。
- 全对全无序对数：`C(112,2)=6216`（S1131）+ `C(32,2)=496`（M1101）= **6712** 对；
- 另加 144 次自身对齐取长度；合计 **6856 次 US-align 调用**。
- 采用“先算全对全矩阵、再按折取最大匹配”的策略，折级暴露无需重复比对（矩阵可复用）。
- 实测资源（本机 224 核 / 2 TB 内存）：
  - M1101：32 并行，约 40 秒；
  - S1131：64 并行，约 56 秒；
  - 单次 US-align 约 0.5–2 秒 CPU，内存占用很低；**0 个失败**。

---

## 附：预检目录清单

```
structure_audit_20261105/
├── manifests/
│   ├── structure_manifest.csv        # 144 个复合物：文件、链、记录数、伙伴、HM
│   └── chain_mapping_audit.csv       # 2232 条记录：链映射、MUT 映射、状态
├── precheck/
│   ├── csv_integrity_report.csv      # 175 个 fold×seed 单元的完整性
│   ├── csv_integrity_summary.json    # 按划分/种子的汇总与独特性
│   ├── inventory_summary.json        # 结构文件清点
│   └── anomalies.csv                 # 异常清单（本次为空）
├── scripts/                          # 全部可复现脚本
├── logs/                             # 运行日志
└── usalign_bin/                      # US-align 二进制副本、provenance、readme
```

**预检异常数：0。**
