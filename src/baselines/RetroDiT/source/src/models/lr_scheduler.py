from torch.optim.lr_scheduler import LambdaLR

def polynomial_lr_schedule(step, total_steps, warmup_steps, warmup_init_lr, lr, lr_end, power):
    if step < warmup_steps:
        return warmup_init_lr + (lr - warmup_init_lr) * step / warmup_steps
    elif step > total_steps:
        return lr_end
    else:
        return lr_end + (lr - lr_end) * (1 - (step - warmup_steps) / (total_steps - warmup_steps)) ** power


class PolyNomialLRScheduler(LambdaLR):
    def __init__(
        self,
        optimizer,
        total_steps: int = 1000,
        warmup_steps: int = 0,
        lr: float = 5e-04,
        lr_end: float = 1e-07,
        warmup_init_lr: float = 1e-07,
        power: float = 1.0,
    ) -> None:

        self.warmup_init_lr = warmup_init_lr
        self.warmup_steps = warmup_steps

        def lr_lambda(step):
            return polynomial_lr_schedule(
                step, total_steps, warmup_steps, warmup_init_lr, lr, lr_end, power
            ) / lr

        super().__init__(optimizer, lr_lambda=lr_lambda)