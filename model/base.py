from typing import Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect
from utils.losses import IOU_loss, dice_loss, diff_loss, keep_loss, lead_loss, simple_loss
from utils.utils import filter as filter_
from utils.metrics import Accuracy, Dice, Recall, Iou, mIoU


class LossWrap:
    def __init__(self, losses):
        self.losses = losses
    
    def __call__(self, x, optimer):
        def _closure(pred):
            loss = {}
            optimer.zero_grad()
            mask = pred['mask']
            loss_total = sum([v['loss'](mask, x.float(), **v['args']) * v.get('weight', 1) for _, v in self.losses.items()])
            loss['total'] = loss_total.item()
            loss_total.backward()
            optimer.step()
            return loss
        return _closure
    
class DeepLossWrap(LossWrap):
    def __init__(self, losses):
        super(DeepLossWrap, self).__init__(losses)

    def __call__(self, x, optimer):
        def _closure(pre:dict[str, Union[torch.Tensor, list[torch.Tensor], tuple[torch.Tensor]]]):
            loss = {}
            mask = pre['mask']
            hidden = pre['hidden']
            optimer.zero_grad()
            loss_total = sum([
                (
                    v['loss'](mask, x.float(), **v['args']) * v.get('weight', 1) + \
                    sum([v['loss'](h, x.float(), **v['args']) * v.get('weight', 1) for h in hidden])
                ) / (len(hidden) + 1)
                for _, v in self.losses.items()])
            loss['total'] = loss_total.item()
            loss_total.backward()
            optimer.step()
            return loss
        return _closure
    
wrap_bce = LossWrap({
    'bce':{
        'loss':F.binary_cross_entropy_with_logits,
        'args':{}
    }
})    

wrap_dice = LossWrap({
    'dice':{
        'loss':dice_loss,
        'args':{}
    }
})

wrap_iou = LossWrap({
    'iou':{
        'loss':IOU_loss,
        'args':{}
    }
})

wrap_keep = LossWrap({
    'keep':{
        'loss':keep_loss,
        'args':{
                'pos_weight': [1.1]
            }
    }
})

wrap_simple = LossWrap({
    'simple':{
        'loss':simple_loss,
        'args':{
                'step': 2
            }
    }
})

wrap_hybrid = LossWrap({
    'iou':{
        'loss':IOU_loss,
        'args':{},
        'weight':0.6
    },
    'diff':{
        'loss':diff_loss,
        'args':{
            'gamma': 0.6
        },
        'weight':0.4
    }
})

def wrap_nl(x, optimer):
    def _closure(pre):
        loss = {}
        mask = pre['mask']
        inc = pre['inc']
        optimer.zero_grad()
        loss_keep = lead_loss(mask, x.float(), inc)
        loss_diff = diff_loss(mask, x.float())
        loss_total = loss_keep * 0.2 + loss_diff * 0.8
        loss['keep'] = loss_keep.item()
        loss['diff'] = loss_diff.item()
        loss['total'] = loss_total.item()
        loss_total.backward()
        optimer.step()
        return loss
    return _closure


class BaseModel(nn.Module):
    ms = {
        'Acc': Accuracy,
        'Iou': Iou,
        'F1': Dice,
        'Recall': Recall,
        'mIoU': mIoU,
    }
    def __init__(self, trace_time=False, as_layer=False):
        super(BaseModel, self).__init__()
        frame = inspect.currentframe().f_back
        arg_names = inspect.getargvalues(frame).args
        self._init_args = {name: frame.f_locals[name] for name in arg_names if name != 'self'}
        self.pre = None
        self.trace_time = trace_time
        self.as_layer = as_layer

    def __call__(self, *args: torch.Any, **kwds: torch.Any) -> torch.Any:
        res = self._wrapped_call_impl(*args, **kwds)
        if isinstance(res, torch.Tensor):
            self.pre = (self.pre if isinstance(self.pre, dict) else {}) | {'mask': res}
        else:
            self.pre = res
        return res

    def get_init_args(self):
        return self._init_args
    
    predict = ...

    default_closure = wrap_bce

    def backward(self, x, optimer, closure:LossWrap=None, clear_stored=True):
        assert self.pre, "Please call forward first"
        if closure is None:
            closure = self.default_closure(x, optimer)
        else:
            closure = closure(x, optimer)
        pre = self.pre
        if clear_stored:self.clear_pre()
        with torch.enable_grad():
            return closure(pre)
    
    def clear_pre(self):
        self.pre = None
    
    def metrics(self, img, mask, with_filtered=True, metrics=None):
        ms = metrics if metrics is not None else self.ms
        pred = self.__call__(img) if self.pre is None else self.pre['mask']
        mask = mask.clone().detach()
        metrics = {}
        pred = pred.clone().detach()
        pred[pred>=0.5]=1
        pred[pred<0.5]=0
        if with_filtered:filtered = torch.tensor(filter_(pred.squeeze().cpu().numpy()), device=pred.device).unsqueeze(0).unsqueeze(0)
        for k, v in ms.items():
            metrics[k] = v(pred, mask)
            if with_filtered:metrics[f'F_{k}'] = v(filtered, mask)
        return metrics
    
    def memo(self):
        return """
        BaseModel for segmentation.
        """
