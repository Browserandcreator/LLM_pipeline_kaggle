import os, random, numpy as np, torch, math

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path

class CosineLRScheduler:
    def __init__(self, optimizer, max_steps, warmup_steps=0, min_lr=0.0, base_lr=None):
        self.optimizer = optimizer
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr
        self.step_num = 0
        if base_lr is None:
            # infer from optimizer
            base_lr = min(g['lr'] for g in optimizer.param_groups)
        self.base_lr = base_lr

    def step(self):
        self.step_num += 1
        if self.step_num <= self.warmup_steps and self.warmup_steps > 0:
            lr = self.base_lr * self.step_num / self.warmup_steps
        else:
            t = (self.step_num - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
            t = min(max(t, 0.0), 1.0)
            lr = self.min_lr + 0.5*(self.base_lr - self.min_lr)*(1 + math.cos(math.pi * t))
        for g in self.optimizer.param_groups:
            g['lr'] = lr
        return lr
