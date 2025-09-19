import argparse, os, json, time, yaml, math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from utils import set_seed, ensure_dir, CosineLRScheduler
from llm.data_text import build_dataloaders
from llm.model_gpt import TinyGPT

def evaluate(model, loader, device, amp=False):
    model.eval()
    ce_total, tok_total = 0.0, 0
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(device), y.to(device)
            with autocast(enabled=amp):
                _, loss = model(x, y)
            bs, T = x.size()
            ce_total += loss.item() * bs * T
            tok_total += bs * T
    return {"val_loss": ce_total / max(tok_total,1), "val_ppl": math.exp(ce_total / max(tok_total,1))}

def build_model(cfg, vocab_size):
    mcfg = cfg["model"]
    return TinyGPT(
        vocab_size=vocab_size,
        n_layer=mcfg.get("n_layer", 6),
        n_head=mcfg.get("n_head", 8),
        n_embd=mcfg.get("n_embd", 512),
        block_size=mcfg.get("block_size", 256),
        dropout=mcfg.get("dropout", 0.1),
        tie_weights=mcfg.get("tie_weights", True),
    )

def train(cfg):
    set_seed(cfg.get("seed", 42))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = cfg["train"].get("amp", True)
    grad_accum = cfg["train"].get("grad_accum_steps", 1)
    max_steps = cfg["train"].get("max_steps", None)

    tokenizer, train_loader, val_loader = build_dataloaders(cfg)

    # 估算词表大小（char/bpe_simple 为 stoi 长度；hf 通过 encode 推断）
    # 这里从一个小字符串推进一遍，确保 embedding 大小正确
    tmp_vocab_size = max(tokenizer.pad_id, tokenizer.eos_id) + 1
    tmp_vocab_size = max(tmp_vocab_size, 4096)  # 下限，防止太小；实际不会影响 hf 的使用
    model = build_model(cfg, vocab_size=cfg["tokenizer"].get("vocab_size", tmp_vocab_size)).to(device)

    if cfg["train"].get("compile", False):
        model = torch.compile(model)  # 需要 PyTorch 2.x

    lr = cfg["train"]["lr"]
    wd = cfg["train"].get("weight_decay", 0.1)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    # 估计训练总步数（用于 Cosine LR）
    est_steps = cfg["train"].get("epochs", 1) * math.ceil(len(train_loader) / grad_accum)
    total_steps = max_steps or est_steps
    scheduler = CosineLRScheduler(optimizer, max_steps=total_steps, warmup_steps=cfg["train"].get("warmup_steps", 1000), base_lr=lr, min_lr=cfg["train"].get("min_lr", 0.0))

    out_dir = ensure_dir(cfg["train"].get("out_dir", "/kaggle/working/outputs"))
    scaler = GradScaler(enabled=amp)
    best = {"val_loss": float("inf")}

    global_step = 0
    for epoch in range(1, cfg["train"].get("epochs", 1)+1):
        model.train()
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}")
        optimizer.zero_grad(set_to_none=True)

        for it, (x, y) in pbar:
            x, y = x.to(device), y.to(device)
            with autocast(enabled=amp):
                _, loss = model(x, y)
                loss = loss / grad_accum

            scaler.scale(loss).backward()

            if (it + 1) % grad_accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                lr_now = scheduler.step()
                global_step += 1

                pbar.set_postfix(loss=f"{loss.item()*grad_accum:.4f}", lr=f"{lr_now:.2e}")

                if max_steps and global_step >= max_steps:
                    break

        metrics = evaluate(model, val_loader, device, amp=amp)
        if metrics["val_loss"] < best["val_loss"]:
            best.update(metrics)
            torch.save(model.state_dict(), os.path.join(out_dir, "model_best.pth"))
        # always save last
        torch.save(model.state_dict(), os.path.join(out_dir, "model_last.pth"))
        with open(os.path.join(out_dir, "metrics.json"), "w") as f:
            json.dump({"best": best, "last": metrics, "epoch": epoch, "global_step": global_step}, f, indent=2)
        print(f"[Epoch {epoch}] val_loss={metrics['val_loss']:.4f}, ppl={metrics['val_ppl']:.2f}")

        if max_steps and global_step >= max_steps:
            break

    print(f"Saved artifacts to: {out_dir}")

def main(cfg_path):
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    train(cfg)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config")
    args = ap.parse_args()
    main(args.config)
