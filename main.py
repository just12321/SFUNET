import os
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
import time
import torch
from torch.utils.data import DataLoader, Subset, random_split
import torch.nn.functional as F
from utils.dataset import CIDataset
from model.utils import init_weights
from utils.Accumulator import Accumulators
from utils.general import TQDM_BAR_FORMAT, colorstr
from sklearn.model_selection import KFold
from utils.utils import apply_config, init_config, overlay_mask, print_config, record
from utils.log import logger
from utils.modelsmanager import ModelsManager
from tqdm import tqdm
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from albumentations import(
    Compose, ShiftScaleRotate, ColorJitter, RandomCrop, Flip, Resize
)
USE_DEFAULT_LOSS = None

def init_model(pretrained):
    if pretrained:
        ckpt = torch.load(config.weights, map_location='cpu')  
        csd = ckpt['model']
        model.load_state_dict(csd, strict=False)  
        logger.info(f'Transferred {len(csd)}/{len(model.state_dict())} items from {config.weights}')  
    else:
        model.apply(init_weights)

def init():
    global config, model, augumentator, sw, optimer, lr_scheduler, device, pretrained
    config = init_config()
    config.seed = None
    config.num_epoch = 500
    config.batch_size = 8
    config.lr = 5e-4
    config.train_metrics_interval = 10
    config.expName = 'Dataset_512_512_' 
    config.mainFile = (script_path:= os.path.realpath(__file__))
    config.trainPath = r'dataset/Dataset/train'
    config.valPath = None
    config.dataset = 'Dataset'
    config.width = 512
    config.height = 512
    config.valid_width = 512
    config.valid_height = 512
    config.vision_period = 10
    config.n_channels = 3
    config.n_classes = 1
    config.wrap_loss = USE_DEFAULT_LOSS
    config.weights = None

    models = ModelsManager('./model')
    modelinfo = models.get_model('sfunet')
    config.modelFile = os.path.join(os.path.dirname(script_path), modelinfo['path'])

    args={
        'in_channels': config.n_channels,
        'num_classes': config.n_classes,
        'num_heads': 1,
        'use_gfg': True,
        'use_sfg': True,
    }

    model = modelinfo['class'](**args)

    config.args = model.get_init_args()
    config.modelName = model.__class__.__name__
    config.expRoot = Path(f'./exp/KFold')
    apply_config(config)
    # logger.addHandler(TqdmHandler())

    pretrained = config.weights and config.weights.endswith('.pt')

    sw = SummaryWriter(config.logPath)

    optimer = torch.optim.Adam(model.parameters(), lr=config.lr)
    lr_scheduler = None
    config.optimizer = optimer

    logger.info(colorstr('red', 'bold', f'{model.__class__.__name__}: ' + model.memo()+'Training on dataset.'))
    print_config(config)

    augumentator = Compose([
            # Resize(config.height, config.width) """When the original resolution of the image is different from the training resolution, it is best to resize it first"""
            ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=180, p=0.8),
            RandomCrop(width=config.width, height=config.height),
            ColorJitter(brightness=(0.8,1.2), contrast=(0.8,1.2), saturation=(0.8,1.2), hue=(-0.1,0.1), p=0.8),
        ])

    device = config.device
    model = model.to(device)

    logger.info(colorstr('yellow', 'bold', 'Model summary:'))
    logger.info(summary(model, (config.n_channels, config.height, config.width), device=device, verbose=0))
    logger.info(colorstr('yellow', 'bold', 'Model info:'))
    logger.info(model)

def train_lead():
    dataset = CIDataset(config.mixPath, augmentator=augumentator, size=(config.width, config.height))
    train_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    train_and_evaluate(-1, train_loader, None)


def get_dataset():
    if config.dataset == 'DCA1':
        def pattern(stem, _): return stem
        return CIDataset(config.trainPath, augmentator=augumentator, size=(config.width, config.height), img_type='png', pattern=pattern),\
               CIDataset(config.valPath, augmentator=augumentator, size=(config.valid_width, config.valid_height), is_train=False, img_type='png', pattern=pattern)
    if config.dataset == 'Dataset':
        return CIDataset(config.trainPath, augmentator=augumentator, size=(config.width, config.height)),\
               CIDataset(config.valPath, augmentator=augumentator, size=(config.valid_width, config.valid_height), is_train=False)
    if config.dataset == 'CHUAC':
        def pattern(stem, _): return f'angio{stem}ok'
        return CIDataset(config.trainPath, augmentator=augumentator, size=(config.width, config.height), img_type='png', pattern=pattern),\
               CIDataset(config.valPath, augmentator=augumentator, size=(config.valid_width, config.valid_height), is_train=False, img_type='png', pattern=pattern)

    raise ValueError(f'Unknown dataset name: {config.dataset}')


def cross_validation(k_folds=5, run_terms=3):
    best_accs.reset()
    trainset, valset = get_dataset()
    if k_folds == 1:
        train_loader = DataLoader(trainset, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(valset, batch_size=1, shuffle=False)
        for fold in range(run_terms):
            train_and_evaluate(fold, train_loader, val_loader, update_best=True)
        infoit(best_accs.items())
        valset.augment(False)
    else:
        kfold = KFold(n_splits=k_folds, shuffle=True, random_state=config.seed)

        for fold, (train_ids, val_ids) in enumerate(kfold.split(trainset)):
            logger.info(colorstr('bright_blue', 'bold', f'FOLD {fold}'))

            # Sample elements randomly from a given list of ids, no replacement.
            subtrain = Subset(trainset, train_ids)
            subval = Subset(trainset, val_ids)
            if not config.valPath:
                val_size = len(subval) // 2  
                test_size = len(subval) - val_size  
                subval, subtest = random_split(subval, [val_size, test_size], generator=torch.Generator().manual_seed(config.seed) if config.seed else None)
            else:
                subtest = valset  

            train_loader = DataLoader(subtrain, batch_size=config.batch_size, shuffle=True)
            val_loader = DataLoader(subval, batch_size=1, shuffle=False)
            test_loader = DataLoader(subtest, batch_size=1, shuffle=False)

            # Train and evaluate the model
            train_and_evaluate(fold, train_loader, val_loader, update_best=False)
            test(fold, test_loader, update_best=True)
        infoit(best_accs.items(), 'test')
        # valset.dataset.augment(False)
    sw.close()
    # recordit(next(iter(val_loader)))
        
best_accs = Accumulators()
accs = {
    'loss_accs': Accumulators(),
    'test_accs': Accumulators(),
    'metrics_accs': Accumulators(),
    'train_metrics_accs': Accumulators()
}

def train_one_epoch(epoch, train_loader, fold):
    sw.add_scalar('Lr', optimer.param_groups[0]['lr'], epoch)
    model.train()
    for batch_idx, (img, mask) in (bar:=tqdm(enumerate(train_loader), total=len(train_loader), bar_format=TQDM_BAR_FORMAT)):
        model.clear_pre()
        img, mask = img.to(device), mask.to(device)
        fake = model(img)
        loss = model.backward(mask, optimer, config.wrap_loss, clear_stored=False)

        accs['loss_accs'].update(loss)
        if epoch % config.train_metrics_interval == 0 :
                accs['train_metrics_accs'].update(model.metrics(img, mask, with_filtered=False))
        mem = f"{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G"
        bar.set_description(f"Fold: {fold} Epoch: {epoch+1}/{config.num_epoch} Mem:{mem}") 
        bar.set_postfix(loss)
    logger.info(f"\t{accs['loss_accs']}")
    logger.info(f"\t{accs['train_metrics_accs']}")
    sw.add_scalars('Loss' if fold == -1 else f'Loss/Fold{fold}', accs['loss_accs'].items(), epoch)
    if epoch % config.train_metrics_interval == 0:
        sw.add_scalars('Train_Metrics' if fold == -1 else f'Train_Metrics/Fold{fold}', accs['train_metrics_accs'].items(), epoch)

    if epoch % config.vision_period == 0:
        train_pred = fake[0].sigmoid().cpu()
        train_mask = mask[0].cpu()
        train_img = img[0][0].unsqueeze(0).cpu()
        fake_train = torch.cat([train_img, train_pred, train_mask,(train_pred - train_mask)/2+0.5, (train_mask - train_pred)/2+0.5, (train_mask-(train_mask*train_pred))/2+0.5], dim=-1).unsqueeze(0)
        sw.add_images('Mask_Train' if fold == -1 else f'Mask_Train/Fold{fold}', fake_train, epoch)

def evaluate_one_epoch(epoch, val_loader, fold, type='val'):
    model.eval()
    vdataset = val_loader.dataset
    if isinstance(vdataset, Subset): vdataset = vdataset.dataset
    if isinstance(vdataset, Subset): vdataset = vdataset.dataset
    vdataset.augment(False)
    with torch.no_grad():
        for img, mask in (pbar:=tqdm(val_loader, desc='Eval', colour='blue', bar_format=TQDM_BAR_FORMAT)):
            model.clear_pre()
            img, mask = img.to(device), mask.to(device)
            metrics = model.metrics(img, mask)
            if type == 'val':
                accs['metrics_accs'].update(metrics)
            if type == 'test':
                accs['test_accs'].update(metrics)
            pbar.set_postfix(metrics)
        if type == 'val':
            sw.add_scalars('Val_Metrics' if fold == -1 else f'Metrics/Fold{fold}', accs['metrics_accs'].items(), epoch)
        if epoch >= 0 and epoch % config.vision_period == 0:
            pred = model.pre['mask'].sigmoid().cpu()
            mask = mask.cpu()
            fake = torch.cat([pred, mask, (mask-(mask*pred))/2+0.5], dim=-1)
            sw.add_images('Mask' if fold == -1 else f'Mask/Fold{fold}', fake, epoch)
    vdataset.augment(True)

def checkpoint(epoch, fold, save_best=True, best_score=0):
    ckpt = {
            'fold': fold,
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict() if lr_scheduler else None,
            'config': dict(config),
            'metrics': accs['metrics_accs'].items(),
            'date': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        }
    if save_best:
        metrics_binary = [v for k, v in accs['metrics_accs'].items().items() if not k.startswith('F_')]
        metrics_filtered = [v for k, v in accs['metrics_accs'].items().items() if k.startswith('F_')]
        logger.info(('%11s'*7)%('', '', 'Accuracy', 'IoU', 'F1', 'Recall', 'mIoU'))
        logger.info(('%11s'*2 + '%11.4g'*5)%('', 'Binary', *metrics_binary))
        logger.info(('%11s'*2 + '%11.4g'*5)%('', 'Filtered', *metrics_filtered))
        bs=sum(metrics_binary) + sum(metrics_filtered)
    else:
        bs = -1
    if best_score <= bs: 
        torch.save(ckpt, config.checkpointPath / f'fold_{fold}_best.pt')
    elif (epoch+1)%config.cpt_period==0 or epoch + 1 == config.num_epoch:
        torch.save(ckpt, config.checkpointPath / f'fold_{fold}_epoch_{epoch}.pt')
    torch.save(ckpt, config.checkpointPath / f'fold_{fold}_last.pt')
    del ckpt
    return bs, accs['metrics_accs'].items()

def train_and_evaluate(fold, train_loader, val_loader=None, update_best=True):
    init_model(pretrained)
    if lr_scheduler: lr_scheduler.reset()
    t0 = time.time()
    start_epoch = 0
    best_score = 0
    best_metrics = None
    for epoch in range(start_epoch, config.num_epoch):
        train_one_epoch(epoch, train_loader, fold)
        if not val_loader is None:evaluate_one_epoch(epoch, val_loader, fold)
        bs, metrics = checkpoint(epoch, fold, not val_loader is None, best_score)
        if best_score <= bs:
            best_score, best_metrics = bs, metrics
        for _,v in accs.items(): v.reset()
        if lr_scheduler: lr_scheduler.step()

        elapsed_time = time.time() - t0
        average_epoch_time = elapsed_time / (epoch - start_epoch + 1)
        remaining_time = average_epoch_time * (config.num_epoch - epoch - 1)

        logger.info(f'\n{epoch - start_epoch + 1} epochs completed in {elapsed_time / 3600:.3f} hours. '
                    f'Estimated remaining time: {remaining_time / 3600:.3f} hours.')
    if not val_loader is None:
        if update_best: best_accs.update(best_metrics)
        infoit(best_metrics)

def test(fold, test_loader, update_best=True):
    if not config.checkpointPath.exists(): raise FileNotFoundError('No checkpoint found')
    model.load_state_dict(torch.load(config.checkpointPath / f'fold_{fold}_best.pt')['model'], strict=False)
    evaluate_one_epoch(-1, test_loader, fold, 'test')
    best_score = accs['test_accs'].items()
    if update_best: best_accs.update(best_score)
    infoit(best_score, 'test')
    sw.add_hparams({}, best_score, run_name=model.__class__.__name__, global_step=fold)

def infoit(best_metrics,type='val'):
    color = 'blue' if type == 'val' else 'green'
    metrics_binary = [v for k, v in best_metrics.items() if not k.startswith('F_')]
    metrics_filtered = [v for k, v in best_metrics.items() if k.startswith('F_')]
    logger.info(colorstr(color, 'bold', 'Best Val Metrics:' if type == 'val' else 'Test Metrics:'))
    logger.info(colorstr(color, 'bold', ('%11s'*6)%('', 'Accuracy', 'IoU', 'F1', 'Recall', 'mIoU')))
    logger.info(colorstr(color, 'bold', ('%11s' + '%11.4g'*5)%('Binary', *metrics_binary)))
    logger.info(colorstr(color, 'bold', ('%11s' + '%11.4g'*5)%('Filtered', *metrics_filtered)))
    logger.info("Waiting for record hidden states...")

def recordit(batch_data):
    model.load_state_dict(torch.load(config.checkpointPath / 'best.pt')['model'], strict=False)
    cam = {}
    sw = SummaryWriter(config.visionPath)
    with record(model, sw, 9, save_cam=cam):
        img, mask = batch_data
        img, mask = img.to(device), mask.to(device)
        pred = model(img)
        loss = F.binary_cross_entropy_with_logits(pred, mask.float())
        loss.backward()
    for k, v in cam.items():
        try:
            grad_cam = F.interpolate(v, size=img.shape[2:], mode='bicubic', align_corners=False).view(img.shape[2:])
            heat_map = overlay_mask(img.squeeze(0).permute(1, 2, 0).cpu(), grad_cam, alpha=0.5)
            sw.add_image(f"heatmap/{k}", heat_map, 0, dataformats='HWC')
        except:
            pass
        finally:
            continue
    cam.clear()
    sw.add_graph(model, img)
    sw.close()

if __name__ == '__main__':
    init()
    cross_validation(5)
