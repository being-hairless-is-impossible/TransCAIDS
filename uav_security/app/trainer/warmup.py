from torch import optim


class WarmupScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, base_lr, final_lr, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.base_lr = base_lr
        self.final_lr = final_lr
        super(WarmupScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        # Apply warmup for the first few epochs
        if self.last_epoch < self.warmup_epochs:
            scale_factor = (self.final_lr - self.base_lr) / self.warmup_epochs
            return [self.base_lr + scale_factor * self.last_epoch for _ in self.base_lrs]
        return [self.final_lr for _ in self.base_lrs]
