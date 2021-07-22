import numpy as np
import torch
import torch.optim
import torch.utils.data
import torch.optim as optim

class base_warmup():
    def __init__(self,optimizer,warm_step,warm_lr,dst_lr):
        ''' base class for warmup scheduler
        Args:
            optimizer: adjusted optimizer
            warm_step: total number of warm_step,(batch num)
            warm_lr: start learning rate of warmup
            dst_lr: init learning rate of train stage eg. 0.01
        '''
        assert warm_lr<dst_lr ,"warmup lr must smaller than init lr"
        self.optimizer=optimizer
        self.warm_lr=warm_lr
        self.init_lr=dst_lr
        self.warm_step=warm_step
        self.stepped=0
        if self.warm_step:
            self.optimizer.param_groups[0]['lr']=self.warm_lr
        
    def step(self):
        self.stepped+=1
        
    def if_in_warmup(self)->bool:
        return True if self.stepped<self.warm_step else False  
    
    def set2initial(self):
        ''' Reset the learning rate to initial lr of training stage '''
        self.optimizer.param_groups[0]['lr']=self.init_lr
    @property
    def now_lr(self):
        return self.optimizer.param_groups[0]['lr']


class const_warmup_scheduler(base_warmup):
    def __init__(self,optimizer,warm_step,warm_lr,dst_lr):
        super().__init__(optimizer,warm_step,warm_lr,dst_lr)
        
    def step(self):
        if(not self.stepped<self.warm_step): return False
        self.optimizer.param_groups[0]['lr']=self.warm_lr
        super().step()
        return True

class linear_warmup_scheduler(base_warmup):
    def __init__(self, optimizer, warm_step, warm_lr, dst_lr):
        super().__init__(optimizer, warm_step, warm_lr, dst_lr)
        if(self.warm_step <= 0):
            self.inc=0
        else:
            self.inc=(self.init_lr-self.warm_lr)/self.warm_step
        
    def step(self)->bool:
        if (not self.stepped<self.warm_step): return False
        self.optimizer.param_groups[0]['lr']+=self.inc
        super().step()
        return True
    
    def still_in_warmup(self)->bool:
        return True if self.stepped<self.warm_step else False

class exponential_warmup_scheduler(base_warmup):
    def __init__(self,optimizer,warm_step,warm_lr,dst_lr):
        super().__init__(optimizer,warm_step,warm_lr,dst_lr)
        
    def step(self):
        ''''''
        super().step()
        raise NotImplementedError
