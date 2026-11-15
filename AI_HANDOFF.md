# 给接收服务器 AI 的交接说明

## 用户已确定的任务

你需要在接收服务器上完成本包内**尚未完成的主表随机种子实验**。服务器共有 **32 张 GPU**，请充分利用，但严格遵循：

**先完成全部剩余训练 → 检查全部最终模型 → 再做主表评测。**

用户已授权这项训练和评测，请在检查实际硬件、数据和模型后推进。不要另外增加实验，不要仅输出计划后停下。遇到确实缺少的权限、模型访问凭据或资源冲突，再向用户说明具体缺项。

这份文件是交接任务和实施说明。**当前包的训练入口仍是单队列，不代表 32 卡并行调度已经实现。你需要先完成下面指出的小范围调度改造。**

## 1. 范围：只完成 40 个最终模型

唯一任务依据：`migration/pending_main_jobs.json`。它包含模型、方法、seed、全部训练参数和 TIS 依赖关系；路径中的 `${PROJECT_ROOT}` 由启动器展开。

| 模型 | seed | 待训练方法 | 最终模型数 |
|---|---|---|---:|
| Llama-3.1-8B Base | 44 | TDPO、TIS-DPO、ConfPO、SFT | 4 |
| Llama-3.1-8B-Instruct | 43、44 | DPO、IPO、CPO、KTO、SimPO、TDPO、TIS-DPO、ConfPO、SFT | 18 |
| Mistral-7B-Instruct-v0.1 | 43、44 | 同上 9 种方法 | 18 |
| 合计 | | | **40** |

另外 10 个 TIS 前向/反向辅助训练和 5 个权重计算，是上述 5 个 TIS-DPO 模型的必需步骤。因此是 **55 个执行任务，40 个最终模型**。

不再训练 Qwen、不再训练 LES-DPO，不做 β 扫描、梯度保护、token 选择等消融，不追加 Mistral Base，不做 HH-RLHF 或通用能力评测。这些均不在此包的执行范围内。

源服务器未完成任务的中间 checkpoint 没有迁移，相关任务应从基础模型重新训练。已完成实验也没有作为权重复制过来，不能凭旧完成记录伪造 `_SUCCESS`。

## 2. 先检查接收环境

1. 读取根目录 `README.md`，检查 `nvidia-smi`、GPU 型号、每卡显存、互联拓扑、CPU 内存及磁盘。
2. 确认 32 张卡是单机，还是多节点合计。脚本的 `torchrun --standalone` 仅支持单节点；多节点应在各节点本地运行独立 8 卡任务，不能直接把全局 GPU 编号当成本机编号。
3. 只使用分配给本任务的 GPU，不停止其他用户进程。基础模型、最终权重和评测输出的目录应可访问；多节点无共享文件系统时，分别同步代码/输入，并明确最终模型的汇集位置。
4. 40 个 7B/8B 合并权重是约 600 GB 量级，另有基础模型、临时 adapter 和缓存；建议准备约 1 TB 可用磁盘，按实际模型大小复核。并行合并模型会占 CPU 内存和磁盘带宽，也需要检查。

源训练环境：Python 3.12，PyTorch 2.9.1 + CUDA 12.8，Transformers 4.57.6，TRL 0.27.1，PEFT 0.13.2，DeepSpeed 0.19.5，FlashAttention 2.8.3。依赖见 `requirements-training.txt`、`requirements-evaluation.txt`。FlashAttention 的编译环境需要与新服务器匹配；不要未经验证就升级训练库并混合结果。

```bash
cd /新服务器上的项目目录
# 先激活已准备好的 Python 环境。
python scripts/restore_data.py
source env.sh
python scripts/check_bundle.py
```

`restore_data.py` 使用两个压缩分块还原 6 个输入文件，逐文件核对 SHA-256；当前展开数据若已存在且一致，会直接复用。训练用 `ultrafeedback_binarized_cleaned/train.jsonl`，共 60,829 对；SFT 使用其中 chosen 回答，不需要另一份 SFT 数据集。

检查以下权重是否齐全，缺少时按 `model/model_manifest.json` 中记录的模型标识补齐。包中只有占位 README，没有任何模型权重或账户令牌：

```text
model/Meta-Llama-3.1-8B/
model/Meta-Llama-3.1-8B-Instruct/
model/Mistral-7B-Instruct-v0.1/
model/Qwen2.5-14B-Instruct/       # 仅在训练结束后的裁判阶段使用
```

将 config、tokenizer、索引和全部权重分片直接放在这些目录下，不要再套一层 `models/.../snapshots/master/`。如果模型访问要求接收者自己的授权，应使用接收者提供的凭据。

## 3. 32 卡训练：4 组，每组 8 卡

优先用 4 个相互独立的 8 卡分布式训练任务并行。对于 4 个 8 卡节点，每个节点一组；对于单机 32 卡，可参考下面的物理卡分配。不要把所有实验合成同一个 32 卡训练任务，也不要每卡各自训练却不调整有效 batch。

| 组 | 单机 32 卡的物理 GPU | 首批模型与 seed |
|---|---|---|
| A | 0–7 | Llama Instruct，seed=43 |
| B | 8–15 | Llama Instruct，seed=44 |
| C | 16–23 | Mistral Instruct，seed=43 |
| D | 24–31 | Mistral Instruct，seed=44 |

先空闲的一组再领取 Llama Base seed=44 的 4 个剩余方法。可以采用依赖图任务队列改善负载均衡，但必须保证每个任务只领取一次，并持久记录其分组。TIS 顺序是：

```text
contrast_forward ─┐
                  ├→ tis_weights → tis_dpo
contrast_reverse ─┘
```

### 必须先处理现有启动器的限制

当前 `scripts/run_pending_seeds.py`：

- 支持 `--gpus`、`--models`、`--methods`、`--execute`。
- **还没有 `--seeds`、`--queue-id` 或独立状态目录选项。**
- 使用同一个 `ICLR/experiments/migration_queue/queue.lock`，所以直接启动四份不会形成四路并行。
- 将 GPU 分配、命令和代码指纹记录进任务 identity；开始后随意换组、改代码会使恢复校验不一致。

请先在接收服务器完成最小调度改造，例如加入 `--seeds` 和 `--queue-id`，各组独立保存状态，再对每个任务增加共享的独占锁/领取记录。也可以实现一个统一的 32 卡总调度器，复用现有 `configure`、`build_command`、`selected_jobs`、`check_completed`、`smaller_batch`。不要改训练损失。

需要保证：

1. 按 `main_result_training` 选取任务后，递归加入其 TIS 依赖；筛选 seed 也必须作用到依赖链。
2. 四组任务互斥；同一个 `(model_key, method, seed)` 不得重复训练或同时写同一输出目录。
3. 不通过反复编辑 `pending_main_jobs.json` 来切组；它有全文件指纹，运行中改动会破坏其他组的恢复校验。
4. 每组的状态、日志、端口及临时缓存应独立。`torchrun --standalone` 现有方式可保留；若改用固定 rendezvous 端口，端口必须互不冲突。
5. 任务启动前固定并记录 GPU 分配和配置。重启时先检查已完成任务的 `_SUCCESS`、配置和所有最终分片，满足条件则跳过；不要只看文件夹是否存在。
6. 不能只删掉全局锁就让四份队列同时跑完整清单。
7. 所有调度改造、配置检查和小规模验证在正式批量运行前完成，不要运行中修改共享源文件。

当前可以立即用于预览的命令是：

```bash
python scripts/run_pending_seeds.py --gpus 0,1,2,3,4,5,6,7
```

分组参数必须由你实现后才使用，不能把本说明中建议的新参数当作现有功能。

## 4. 保持原参数，OOM 时保持有效 batch

每个任务的有效 batch 必须从清单计算：

```text
原有效 batch = 原 GPU 数 × 原每卡 batch × 原梯度累积
新有效 batch = 新 GPU 数 × 新每卡 batch × 新梯度累积
```

二者应相同。原有 `configure()` 已能完成 4 卡到 8 卡的转换。按当前清单，8 卡时主要配置如下：

| 方法/模型 | 每卡 batch | 梯度累积 | 有效 batch |
|---|---:|---:|---:|
| Llama Instruct 的 DPO | 8 | 4 | 256 |
| Mistral Instruct 的 DPO | 24 | 4 | 768 |
| IPO/CPO/KTO/SimPO/ConfPO | 24 | 4 | 768 |
| TDPO | 12 | 8 | 768 |
| TIS-DPO | 6 | 16 | 768 |
| TIS 前向/反向辅助训练 | 24 | 4 | 768 |
| SFT | 8 | 4 | 256 |

以每条 JSON 的参数为准，不要把所有 DPO 都当成同一有效 batch。TIS 权重计算的 `batch_size` 是推理计算参数，不属于上表的优化器 batch。

GPU 型号未知，不能预先承诺这些 microbatch 都不 OOM。沿用现有处理：检测 CUDA OOM → 将 microbatch 对半降低 → 增加梯度累积保持有效 batch → 留存失败记录后重试。奇数 microbatch 用 `smaller_batch` 选择可整除的更小值；microbatch=1 仍 OOM 则停止并报告具体原因。非 OOM 错误不得伪装成 OOM 重试。

不要自行改变 seed、β、学习率、epoch、数据、长度上限、目标函数、归一化或 LoRA 参数。不要把这些新增种子自动与旧 seed=42 混合统计。

当前训练器最终导出合并权重，未设置可恢复的定期训练 checkpoint；任务中断后不能宣称从原训练步数继续。新服务器重新运行失败任务时，妥善区分失败尝试与有效最终输出。

## 5. 训练阶段验收后才允许评测

总调度器必须检查清单中的 **40 个 main_result_training** 全部完成，确认：

- `_SUCCESS` 存在，`run_config.json` 的模型、方法、seed 与清单一致。
- `model/config.json`、tokenizer 和全部权重分片齐全；不是只有 adapter 或未完成的合并输出。
- TIS 依赖产物有效；没有用失败日志冒充成功。
- 训练进程都正常退出并释放 GPU，再启动评测服务。

输出一份 40 行训练完成清单，带 `model_key / method / seed / checkpoint_path / effective_batch / status`。这份清单用于评测，不要遍历任意旧目录把额外模型加入。

## 6. 训练完成后，评测这 40 个模型

只评测三个主表基准、四项指标：

| 基准 | 每个模型的数据量 | 指标 |
|---|---:|---|
| AlpacaEval2 | 805 条 | LC、WR |
| Arena-Hard v0.1 | 500 条 | WR |
| MT-Bench | 80 组 × 2 轮 | 160 个评分的平均值 |

测试题、所需参考回答、LC 计算数据及全部评分代码已经包含。裁判统一使用 **Qwen2.5-14B-Instruct**，不换成新模型，也不能把本地裁判成绩称为官方 GPT-4 成绩。

### 6.1 先并行生成回答

以模型为单位分配到空闲 GPU，单卡能容纳时可最多同时生成 32 个模型的回答，再处理剩余模型。使用现有入口，给每个模型唯一 alias，例如 `llama_instruct_dpo_seed43`。原始 Base 的模型额外传 `--base`：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_checkpoint.py \
  --checkpoint /某次训练输出/model \
  --alias llama_instruct_dpo_seed43 --family llama3 --stage generate
```

Llama 使用 `--family llama3`，Mistral 使用 `--family mistral`。不要把同一模型的多个生成进程写进同一个输出目录；当前最简单可靠的并行单位是完整模型。

### 6.2 再启动本地 14B 裁判并行评分

候选回答生成完成后释放生成模型，再按实际显存启动独立裁判副本。显存足够时每卡一个 14B、副本 TP=1、显存比例 0.85；如单卡不够，需要在同一节点内用 TP=2/4 并相应减少副本数。不要因卡数多而盲目在显存不足的单卡上加载 14B。

同一主机的各服务使用不同端口，例如 8065、8066……；多节点使用节点地址与对应端口，单机服务默认绑定 127.0.0.1。每个模型的完整评分交给一个裁判端点；不同模型可并行分配。

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve model/Qwen2.5-14B-Instruct \
  --served-model-name qwen-local-judge --host 127.0.0.1 --port 8065 \
  --tensor-parallel-size 1 --gpu-memory-utilization 0.85 \
  --max-model-len 16384 --max-num-seqs 256

python scripts/evaluate_checkpoint.py \
  --checkpoint /某次训练输出/model \
  --alias llama_instruct_dpo_seed43 --family llama3 --stage judge \
  --endpoint http://127.0.0.1:8065/v1 --parallel 256
```

根据 CPU/RAM、KV cache、实际吞吐和错误率调整副本数及并发。OOM 时降低并发、`max_num_seqs` 或批 token 预算，不改变评测温度、最大回答长度或评分规则。多份 AlpacaEval 进程会共享其环境中的资产缓存，批量运行前先串行准备/验证一次，再验证并行结果完整性。

### 固定评测参数

- 被测模型最大新输出：1024 token；生成上下文默认 8192。
- AlpacaEval2 / Arena-Hard 生成温度：0.0。
- MT-Bench：writing/roleplay 为 0.7；STEM/humanities 为 0.1；extraction/math/coding/reasoning 为 0.0；两轮按同一类别设置。
- 裁判温度：0.0；Arena 裁判最多 2048 token，MT 裁判最多 256 token。
- Arena 沿用现有的交换答案顺序评分和结果聚合，不新增 Arena LC。
- AlpacaEval2 LC 沿用包中的长度控制实现，不更换计算方式。

先用 `--limit 2` 验证生成→评分整条链路，确认正常后再完整评测。小样本输出已经与完整评测分目录保存，不能用小样本完成标记替代完整评测。

## 7. 最终交付

1. 40 个最终模型、完整训练配置和可追溯日志。
2. 40 个模型各自的生成回答、裁判结果、四项指标。
3. 汇总成一个 40 行 CSV，字段至少为 `model_key, method, seed, alpaca_lc, alpaca_wr, arena_wr, mt_overall, checkpoint_path`。现有单模型结果位于 `ICLR/evaluation/portable/full/<alias>/summary/four_metrics.csv`，需要由总调度器合并。
4. 核对每个模型的数据 ID/数量：805、500、80×2；API 错误、无效裁判输出不得悄悄当作正常低分。使用已有校验函数，错误处理完再汇总。
5. 给用户报告已完成/失败/剩余数量、GPU 利用情况和耗时估计。耗时依据接收服务器实际吞吐估算，不沿用旧机器时间。

用户没有要求上传模型、发邮件或新增其他实验；本交接只要求完成上述训练与评测。
