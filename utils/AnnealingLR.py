from torch.optim.lr_scheduler import SequentialLR, CosineAnnealingLR, LambdaLR
import math
from bisect import bisect_right

class AnnealingLR(SequentialLR):
    """Annealing Learning Rate Scheduler."""
    def __init__(self, optimizer, epoch_num, T_max, eta_min=0, last_epoch=-1, verbose=False):
        count = math.ceil(epoch_num/T_max)
        super().__init__(optimizer, 
                        [CosineAnnealingLR(optimizer, T_max*(i+1.5), eta_min=eta_min, verbose=verbose) for i in range(count)], 
                        milestones=[T_max*i for i in range(1,count)], 
                        last_epoch=last_epoch,
                        verbose=verbose
                )

    def get_lr(self):
        return self._last_lr
    
    def reset(self):
        self.last_epoch = 0
    
    def step(self):
        self.last_epoch += 1
        idx = bisect_right(self._milestones, self.last_epoch)
        scheduler = self._schedulers[idx]
        scheduler.step(self.last_epoch)
        self._last_lr = scheduler.get_last_lr()

class StepAnnealingLR(SequentialLR):
    """Annealing Learning Rate Scheduler."""
    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1, verbose=False):
        super().__init__(optimizer, 
                        [CosineAnnealingLR(optimizer, T_max, eta_min=eta_min, verbose=verbose), LambdaLR(optimizer, lambda _: eta_min, verbose=verbose)], 
                        milestones=[T_max], 
                        last_epoch=last_epoch,
                        verbose=verbose
                )

    def get_lr(self):
        return self._last_lr
    
    def step(self):
        self.last_epoch += 1
        idx = bisect_right(self._milestones, self.last_epoch)
        scheduler = self._schedulers[idx]
        scheduler.step(self.last_epoch)
        self._last_lr = scheduler.get_last_lr()