# 只继续尚未完成的主表种子实验

接收服务器上的 AI 请先阅读 **[AI_HANDOFF.md](AI_HANDOFF.md)**：使用 32 张卡完成剩余种子实验，全部训练结束后再评测；其中说明了现有单队列入口需要补充的并行调度。

本目录只包含剩余种子训练所需的代码、数据，以及这些模型的三个主表基准评测。不包含已完成实验、LES-DPO 重跑、β 扫描、梯度消融、通用能力评测或旧调度脚本。

| 模型 | 待训练数量 |
|---|---:|
| Llama-3.1-8B Base | 4：seed=44 的 TDPO、TIS-DPO、ConfPO、SFT |
| Llama-3.1-8B-Instruct | 18：9 种方法 × seed=43、44 |
| Mistral-7B-Instruct-v0.1 | 18：9 种方法 × seed=43、44 |
| 合计 | **40** |

9 种方法：DPO、IPO、CPO、KTO、SimPO、TDPO、TIS-DPO、ConfPO、SFT。TIS-DPO 还需 10 个前向/反向辅助训练和 5 个权重计算，因此执行清单是 55 步；它们是上述模型的必要依赖，不是额外实验。清单在 `migration/pending_main_jobs.json`。

## 目录怎么用

```text
scripts/run_pending_seeds.py   唯一训练入口
scripts/evaluate_checkpoint.py 三基准评测入口
src/                          主表训练器及公共模块
ICLR/                         队列工具、生成回答和裁判评分实现
configs/ds_config.json         DeepSpeed 配置
model/                        三个基础模型和 14B 裁判的占位目录
data_bundle/                  迁移用的数据分块、校验清单
```

展开的数据还保留在原相对路径中。训练只用 `ultrafeedback_binarized_cleaned/train.jsonl` 的 60,829 对样本；SFT 同样使用它的 chosen 回答，不需要额外 `train_sft.jsonl`。

## 新服务器准备

Python 3.12；原环境 PyTorch 2.9.1 + CUDA 12.8、FlashAttention 2.8.3。先准备匹配服务器 CUDA 的环境，然后：

```bash
cd /下载后的项目目录
python -m pip install -r requirements-training.txt
python -m pip install packaging ninja
python -m pip install flash-attn==2.8.3 --no-build-isolation
python scripts/restore_data.py
source env.sh
python scripts/check_bundle.py
```

当前交付目录已经展开数据。通过 GitHub 下载后，使用 `restore_data.py` 从分块恢复；不需要重新寻找数据。还原会校验每个文件的 SHA-256。

将基础模型的配置、tokenizer 和全部权重分片直接放进：

```text
model/Meta-Llama-3.1-8B/
model/Meta-Llama-3.1-8B-Instruct/
model/Mistral-7B-Instruct-v0.1/
```

占位 README 不是权重，不要再嵌套 `models/.../snapshots/master/`。没有打包任何模型权重、令牌或邮箱授权码。

## 只训练剩余模型

```bash
source env.sh
# 先预览，不占 GPU：
python scripts/run_pending_seeds.py --gpus 0,1,2,3
# 实际训练：
python scripts/run_pending_seeds.py --gpus 0,1,2,3 --execute
```

使用 8 卡时，把 GPU 列表换成 `0,1,2,3,4,5,6,7`。脚本保持每个任务原有有效 batch；OOM 时降低每卡 batch 并增加梯度累积。可以用 `--models llama_instruct` 或 `--methods dpo ipo` 筛选；TIS 依赖会自动带上。

未携带旧服务器未结束训练的中间 checkpoint，剩余任务在新服务器从基础权重重新训练。此入口只执行上述 40 个最终模型和必要依赖，完成后退出，不接其他实验。新运行的日志、状态及权重在 `ICLR/experiments/`。

## 主表测试数据与代码

测试数据、参考回答和评分代码均包含：

- AlpacaEval2：805 条，输出 LC 和 WR；包含 LC 所需 `df_gamed.csv`。
- Arena-Hard v0.1：500 条，输出 WR。
- MT-Bench：80 组两轮对话，输出平均评分。

评测采用本地 Qwen2.5-14B-Instruct 裁判，不能将结果称为官方 GPT-4 裁判成绩。把裁判放进 `model/Qwen2.5-14B-Instruct/`，然后安装评测依赖：

```bash
python -m pip install -r requirements-evaluation.txt
source env.sh

# 举例：把 checkpoint 路径替换成某次训练输出中的 model/。
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_checkpoint.py \
  --checkpoint /训练输出/model --alias llama_instruct_dpo_seed43 \
  --family llama3 --stage generate

# 另一个终端启动裁判：
CUDA_VISIBLE_DEVICES=1 vllm serve model/Qwen2.5-14B-Instruct \
  --served-model-name qwen-local-judge --host 127.0.0.1 --port 8065 \
  --tensor-parallel-size 1 --gpu-memory-utilization 0.85 \
  --max-model-len 16384 --max-num-seqs 256

python scripts/evaluate_checkpoint.py \
  --checkpoint /训练输出/model --alias llama_instruct_dpo_seed43 \
  --family llama3 --stage judge --parallel 256
```

Mistral 使用 `--family mistral`；Llama Base 额外加 `--base`。同一个模型的两个阶段使用相同参数。输出按 alias 分开保存；均可加 `--limit 2` 做小样本检查。

沿用原评测参数：生成回答最多 1024 token；Alpaca/Arena 温度为 0；MT 按类别为 0/0.1/0.7；裁判温度为 0。训练完成不会自动启动评测，按需调用以上入口。

此副本只进行了文件、CPU 逻辑及依赖检查，没有在目标服务器实际执行 GPU 训练。第三方数据和代码沿用原有许可。
