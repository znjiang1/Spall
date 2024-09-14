import torch
import math

class CosineWarmupDecay(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_step, total_step, multi, print_step, last_epoch=-1, verbose=False):
        self.initial_lr = [group['lr'] for group in optimizer.param_groups]
        self.min_lr = [lr * 0.0001 for lr in self.initial_lr]
        self.warmup_step = warmup_step
        self.total_step = total_step
        self.multi = multi
        self.print_step = print_step
        self.current_step = 0
        super(CosineWarmupDecay, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self.current_step >= self.total_step:  
            self.warmup_step *= (1 + self.multi)
            self.total_step *= (1 + self.multi)
            self.current_step = 0

        decayed_learning_rate = [min_lr + 0.5 * (initial_lr - min_lr) * 
                                 (1 + math.cos(math.pi * (self.current_step-self.warmup_step) / 
                                               (self.total_step-self.warmup_step)))
                                 for initial_lr, min_lr in zip(self.initial_lr, self.min_lr)]

        k = [(initial_lr - min_lr) / self.warmup_step for initial_lr, min_lr in zip(self.initial_lr, self.min_lr)]
        warmup = [k * self.current_step + min_lr for k, min_lr in zip(k, self.min_lr)]

        decayed_learning_rate = [warmup_lr if self.current_step < self.warmup_step else decayed_lr 
                                 for warmup_lr, decayed_lr in zip(warmup, decayed_learning_rate)]


        self.current_step += 1

        return decayed_learning_rate