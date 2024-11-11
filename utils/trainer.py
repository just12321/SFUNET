import os
import time

from sklearn.model_selection import KFold
import torch
from torchsummary import summary
from tqdm import tqdm
from model.base import BaseModel
from torch.utils.tensorboard import SummaryWriter
from model.utils import init_weights
from utils.Accumulator import Accumulators
from utils.general import TQDM_BAR_FORMAT, colorstr
from utils.modelsmanager import ModelsManager
from torch.utils.data import DataLoader, Subset
from utils.log import logger
from utils.utils import apply_config, overlay_mask, record
import torch.nn.functional as F


class Trainer:
    def __init__(self, config):
        self.config = config
        self.setup_environment()
        self.setup_model()
        self.setup_logging()

    def setup_environment(self):
        config = self.config
        self.models = ModelsManager('./model')
        modelinfo = self.models.get_model(config.model)
        config.modelFile = os.path.join(os.path.dirname(config.mainFile), modelinfo['path'])

        args = config.modelArgs

        self.model: BaseModel = modelinfo['class'](**args)
        config.args = self.model.get_init_args()
        config.modelName = self.model.__class__.__name__
        apply_config(config)

        self.pretrained = config.weights and config.weights.endswith('.pt')
        self.sw = SummaryWriter(config.logPath)
        self.optimer = config.optimer(self.model.parameters(), **config.optimerArgs)
        self.lr_scheduler = config.lr_scheduler(self.optimer, **config.lr_schedulerArgs)

    def setup_model(self):
        self.device = self.config.device
        self.num_epoch = self.config.num_epoch
        self.model = self.model.to(self.device)
        self.wrap_loss = self.config.wrap_loss
        self.best_accs = Accumulators()
        self.accs = {
            'loss_accs': Accumulators(),
            'metrics_accs': Accumulators(),
            'train_metrics_accs': Accumulators()
        }
        logger.info(colorstr('yellow', 'bold', 'Model summary:'))
        logger.info(summary(self.model, (self.config.n_channels, self.config.height, self.config.width), device=self.device, verbose=0))
        logger.info(colorstr('yellow', 'bold', 'Model info:'))
        logger.info(self.model)

    def init_model(self, pretrained):
        if pretrained:
            ckpt = torch.load(self.config.weights, map_location='cpu')
            csd = ckpt['model']
            self.model.load_state_dict(csd, strict=False)
            logger.info(f'Transferred {len(csd)}/{len(self.model.state_dict())} items from {self.config.weights}')
        else:
            self.model.apply(init_weights)

    def train_lead(self):
        self.train_and_evaluate(-1, self.train_loader, None)
        self.sw.close()

    def cross_validation(self, k_folds=5, run_terms=3):
        self.best_accs.reset()
        if k_folds == 1:
            train_loader = self.train_loader
            val_loader = self.val_loader
            for fold in range(run_terms):
                self.train_and_evaluate(fold, train_loader, val_loader)
            self.infoit(self.best_accs.items())
            valset.augment(False)
        else:
            dataset = self.train_dataset
            kfold = KFold(n_splits=k_folds, shuffle=True, random_state=self.config.seed)

            for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
                logger.info(colorstr('bright_blue', 'bold', f'FOLD {fold}'))

                # Sample elements randomly from a given list of ids, no replacement.
                trainset = Subset(dataset, train_ids)
                valset = Subset(dataset, val_ids)

                train_loader = DataLoader(trainset, batch_size=self.config.batch_size, shuffle=True)
                val_loader = DataLoader(valset, batch_size=1, shuffle=False)

                # Train and evaluate the model
                self.train_and_evaluate(fold, train_loader, val_loader)
            self.infoit(self.best_accs.items())
            valset.dataset.augment(False)
        self.sw.close()
        self.recordit(next(iter(val_loader)))

    def train_one_epoch(self, epoch, train_loader, fold):
        self.sw.add_scalar('Lr', self.optimer.param_groups[0]['lr'], epoch)
        self.model.train()
        for batch_idx, (img, mask) in (bar := tqdm(enumerate(train_loader), total=len(train_loader), bar_format=TQDM_BAR_FORMAT)):
            self.model.clear_pre()
            img, mask = img.to(self.device), mask.to(self.device)

            fake = self.model(img)
            loss = self.model.backward(mask, self.optimer, self.wrap_loss, clear_stored=False)

            self.accs['loss_accs'].update(loss)
            if epoch % self.config.train_metrics_interval == 0:
                self.accs['train_metrics_accs'].update(self.model.metrics(img, mask, with_filtered=False))
            mem = f"{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G"
            bar.set_description(f"Fold: {fold} Epoch: {epoch+1}/{self.num_epoch} Mem:{mem}")
            bar.set_postfix(loss)
        logger.info(f"\t{self.accs['loss_accs']}")
        logger.info(f"\t{self.accs['train_metrics_accs']}")
        self.sw.add_scalars('Loss' if fold == -1 else f'Loss/Fold{fold}', self.accs['loss_accs'].items(), epoch)
        if epoch % self.config.train_metrics_interval == 0:
            self.sw.add_scalars('Train_Metrics' if fold == -1 else f'Train_Metrics/Fold{fold}', self.accs['train_metrics_accs'].items(), epoch)

        if epoch % self.config.vision_period == 0:
            train_pred = fake[0].sigmoid().cpu()
            train_mask = mask[0].cpu()
            train_img = img[0][0].unsqueeze(0).cpu()
            fake_train = torch.cat([train_img, train_pred, train_mask, (train_pred - train_mask) / 2 + 0.5, (train_mask - train_pred) / 2 + 0.5, (train_mask - (train_mask * train_pred)) / 2 + 0.5], dim=-1).unsqueeze(0)
            self.sw.add_images('Mask_Train' if fold == -1 else f'Mask_Train/Fold{fold}', fake_train, epoch)

    def evaluate_one_epoch(self, epoch, val_loader, fold):
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (img, mask) in (bar := tqdm(enumerate(val_loader), total=len(val_loader), bar_format=TQDM_BAR_FORMAT)):
                img, mask = img.to(self.device), mask.to(self.device)

                fake = self.model(img)
                loss = self.model.backward(mask, None, self.wrap_loss, clear_stored=False)

                self.accs['loss_accs'].update(loss)
                self.accs['metrics_accs'].update(self.model.metrics(img, mask, with_filtered=False))
                mem = f"{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G"
                bar.set_description(f"Fold: {fold} Epoch: {epoch+1}/{self.num_epoch} Mem:{mem}")
                bar.set_postfix(loss)

            logger.info(f"\t{self.accs['loss_accs']}")
            logger.info(f"\t{self.accs['metrics_accs']}")
            self.sw.add_scalars('Loss' if fold == -1 else f'Loss/Fold{fold}', self.accs['loss_accs'].items(), epoch)
            self.sw.add_scalars('Metrics' if fold == -1 else f'Metrics/Fold{fold}', self.accs['metrics_accs'].items(), epoch)

            if self.accs['metrics_accs'].acc_dict['mIoU'] > self.best_accs.acc_dict['mIoU']:
                self.best_accs.update(self.accs['metrics_accs'].acc_dict)
                torch.save({'epoch': epoch, 'model': self.model.state_dict(), 'optimizer': self.optimer.state_dict()}, self.config.bestModelPath)
                logger.info(f"Best Model Saved: {self.config.bestModelPath}")

    def train_and_evaluate(self, fold, train_loader, val_loader):
        self.init_model(self.pretrained)
        self.accs['loss_accs'].reset()
        self.accs['metrics_accs'].reset()
        self.best_accs.reset()
        start_time = time.time()
        for epoch in range(self.num_epoch):
            self.train_one_epoch(epoch, train_loader, fold)
            if val_loader is not None:
                self.evaluate_one_epoch(epoch, val_loader, fold)
        end_time = time.time()
        logger.info(f'Total time: {end_time - start_time:.2f}s')
        logger.info(colorstr('yellow', 'bold', f'Best metrics: {self.best_accs}'))
        self.recordit(next(iter(val_loader)))

    def recordit(self, batch_data):
        self.model.load_state_dict(torch.load(self.config.checkpointPath / 'best.pt')['model'], strict=False)
        cam = {}
        sw = SummaryWriter(self.config.visionPath)
        with record(self.model, sw, 9, save_cam=cam):
            img, mask = batch_data
            img, mask = img.to(self.device), mask.to(self.device)
            pred = self.model(img)
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
        sw.add_graph(self.model, img)
        sw.close()

    def infoit(self, best_metrics:Accumulators):
        metrics_binary = [v for k, v in best_metrics.items() if not k.startswith('F_')]
        metrics_filtered = [v for k, v in best_metrics.items() if k.startswith('F_')]
        logger.info(colorstr('green', 'bold', 'Best Metrics:'))
        logger.info(colorstr('green', 'bold', ('%11s'*6)%('', 'Accuracy', 'IoU', 'F1', 'Recall', 'mIoU')))
        logger.info(colorstr('green', 'bold', ('%11s' + '%11.4g'*5)%('Binary', *metrics_binary)))
        logger.info(colorstr('green', 'bold', ('%11s' + '%11.4g'*5)%('Filtered', *metrics_filtered)))
        logger.info("Waiting for record hidden states...")