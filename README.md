# Kaggle + PyTorch LLM（Causal LM）训练模板

面向 **Kaggle GPU** 的轻量级 GPT 训练脚手架：支持 **自带简易分词器（离线）** 与 **HuggingFace 分词器（在线）** 两种模式，
训练目标为 **自回归下一词预测（CrossEntropy）**。

## 目录结构
```
your-repo/
├─ src/
│  ├─ llm/
│  │  ├─ data_text.py          # 文本数据集与数据块化（char/BPE两种模式）
│  │  ├─ tokenizer_simple.py   # 离线可用的字符级/简易BPE分词器
│  │  └─ model_gpt.py          # 轻量 GPT 解码器（多头注意力 + MLP）
│  ├─ train_llm.py             # 训练入口（读取 YAML 配置，AMP + 累积步 + Cosine LR）
│  └─ utils.py                 # 随机种子、目录、学习率计划
├─ configs/
│  ├─ llm_base.yaml            # 通用超参（上下文长度、层数、头数、维度等）
│  └─ llm_kaggle_t4.yaml       # Kaggle T4 友好配置（小模型、显存更稳）
├─ scripts/
│  └─ run_llm_local.sh         # 本地一键跑
├─ requirements.txt
├─ .gitignore
└─ README.md
```

## 快速开始（本地）
```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip wheel
pip install -r requirements.txt
python src/train_llm.py --config configs/llm_kaggle_t4.yaml
```

## 在 Kaggle Notebook 上运行
1. 右侧 **Add-ons → Secrets** 新建 `GITHUB_TOKEN`；开启 GPU；在 Data 面板添加你的文本 Dataset（例如一个 .txt 或多个 .txt）。  
2. 第一格：
```python
import os, subprocess, pathlib, json
token = os.environ.get("GITHUB_TOKEN"); assert token
work = "/kaggle/working"; repo = "github.com/YourName/your-repo.git"
subprocess.run(["bash","-lc",f"cd {work} && git clone https://{token}@{repo} project"], check=True)
subprocess.run(["bash","-lc","python -m pip install -U pip wheel"], check=True)
subprocess.run(["bash","-lc","pip install -r /kaggle/working/project/requirements.txt"], check=True)

# 训练（如需HF分词器，把 configs 里 tokenizer.mode 改为 'hf'）
subprocess.run(["bash","-lc","python /kaggle/working/project/src/train_llm.py --config /kaggle/working/project/configs/llm_kaggle_t4.yaml"], check=True)
```
3. 数据路径：把 `configs/*.yaml` 里 `data.text_glob` 指向 `/kaggle/input/<dataset-name>/**/*.txt`。  
4. 结果会写到 `/kaggle/working/outputs/`；需要长期保存就发布为 **Kaggle Dataset 新版本**。

## 两种分词/数据模式
- **离线简易模式**：`tokenizer.mode: "char"` 或 `"bpe_simple"`（不依赖外网，适合 Kaggle 禁网提交）。  
- **HuggingFace 模式**：`tokenizer.mode: "hf"`，指定 `pretrained_name`（如 `gpt2`）。适合交互式 Notebook 有网的情况。

## 训练特性
- 自回归损失（左移标签）
- AMP（自动混合精度）
- 梯度累积（节省显存）
- Cosine 学习率调度 + 线性 warmup
- 可选 torch.compile（PyTorch 2.x）
- 简易梯度检查点（按层 checkpoint）

> 注：这是教学/入门模板，重点在 **结构清晰 + 可复现**。要追求更高性能可考虑 FlashAttention、RoPE、Fused Kernels、ZeRO/FSDP 等。
