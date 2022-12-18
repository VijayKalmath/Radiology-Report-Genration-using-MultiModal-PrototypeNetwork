import logging
import os
from abc import abstractmethod

import torch
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import sigmoid
# import torchmetrics as tm
# from torchmetrics.functional.classification import multilabel_f1_score
from sklearn.metrics import f1_score
import numpy as np 
from numpy import inf
import wandb


class BaseTrainer(object):
    def __init__(self, model, args):
        self.args = args

        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        self.logger = logging.getLogger(__name__)

        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(args.n_gpu)

        # Move Model to GPU
        self.model = model.to(self.device)

        # Set up Criterion 
        self.criterion = BCEWithLogitsLoss()

        # set up optimizer
        self.optimizer = Adam(model.parameters())

        # Set up LR Scheduler 
        self.lr_scheduler = ReduceLROnPlateau(self.optimizer, 'min')

        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        self.epochs = self.args.epochs
        self.save_period = self.args.save_period

        self.mnt_mode = args.monitor_mode
        self.mnt_metric = "val_" + args.monitor_metric
        self.mnt_metric_test = "test_" + args.monitor_metric
        assert self.mnt_mode in ["min", "max"]

        self.mnt_best = inf if self.mnt_mode == "min" else -inf
        self.early_stop = getattr(self.args, "early_stop", inf)

        self.start_epoch = 1
        self.n_gpu = args.n_gpu
        self.checkpoint_dir = args.save_dir

        self.best_recorder = {
            "val": {self.mnt_metric: self.mnt_best},
            "test": {self.mnt_metric_test: self.mnt_best},
        }

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

    @abstractmethod
    def _train_epoch(self, epoch):
        raise NotImplementedError

    def train(self):
        not_improved_count = 0
        best_epoch = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {"epoch": epoch}
            log.update(result)

            improved_val = self._record_best(log)

            # print logged informations to the screen
            for key, value in log.items():
                self.logger.info("\t{:15s}: {}".format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != "off":
                if improved_val:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                    best_epoch = epoch
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info(
                        "Validation performance didn't improve for {} epochs. "
                        "Training stops.".format(self.early_stop)
                    )
                    break
            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)
            print("best performance in epoch: ", best_epoch)

    def _record_best(self, log):
        improved_val = (
            self.mnt_mode == "min"
            and log[self.mnt_metric] <= self.best_recorder["val"][self.mnt_metric]
        ) or (
            self.mnt_mode == "max"
            and log[self.mnt_metric] >= self.best_recorder["val"][self.mnt_metric]
        )
        if improved_val:
            self.best_recorder["val"].update(log)

        improved_test = (
            self.mnt_mode == "min"
            and log[self.mnt_metric_test]
            <= self.best_recorder["test"][self.mnt_metric_test]
        ) or (
            self.mnt_mode == "max"
            and log[self.mnt_metric_test]
            >= self.best_recorder["test"][self.mnt_metric_test]
        )
        if improved_test:
            self.best_recorder["test"].update(log)
        
        return improved_val

    def _prepare_device(self, n_gpu_use):
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning(
                "Warning: There's no GPU available on this machine,"
                "training will be performed on CPU."
            )
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self.logger.warning(
                "Warning: The number of GPU's configured to use is {}, but only {} are available "
                "on this machine.".format(n_gpu_use, n_gpu)
            )
            n_gpu_use = n_gpu
        device = torch.device("cuda:0" if n_gpu_use > 0 else "cpu")
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _save_checkpoint(self, epoch, save_best=False):
        state = {
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "monitor_best": self.mnt_best,
        }
        filename = os.path.join(self.checkpoint_dir, "current_checkpoint.pth")
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = os.path.join(self.checkpoint_dir, "model_best.pth")
            torch.save(state, best_path)
            self.logger.info("Saving current best: model_best.pth ...")


class Trainer(BaseTrainer):
    def __init__(self, model, args, train_dataloader, val_dataloader, test_dataloader):
        super(Trainer, self).__init__(model, args)

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader

    def _train_epoch(self, epoch):
        self.logger.info(
            "[{}/{}] Start to train in the training set.".format(epoch, self.epochs)
        )
        
        # Dictionary to store entries for logging 
        wandb_dictionary = {}

        # Loss value 
        cross_entropy_loss = 0

        self.model.train()
        for batch_idx, (
            _,
            images,
            labels,
        ) in enumerate(self.train_dataloader):

            images, labels = (
                images.to(self.device),
                labels.to(self.device),
            )

            output = self.model(images)

            loss = self.criterion(output, labels.type(torch.float))
            cross_entropy_loss += loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if batch_idx % self.args.log_period == 0:

                self.logger.info(
                    "[{}/{}] Step: {}/{}, Cross Entropy Loss Ls: {:.5f}.".format(
                        epoch,
                        self.epochs,
                        batch_idx,
                        len(self.train_dataloader),
                        cross_entropy_loss / (batch_idx + 1),
                    )
                )

        log = {
            "cross_entropy_loss": cross_entropy_loss.detach().cpu().numpy() / len(self.train_dataloader),
        }

        self.logger.info(
            "[{}/{}] Start to evaluate in the validation set.".format(
                epoch, self.epochs
            )
        )
        self.model.eval()

        with torch.no_grad():
            val_gts, val_res = [], []
            for batch_idx, (
                _,
                images,
                labels,
            ) in enumerate(self.val_dataloader):
                images, labels = (
                    images.to(self.device),
                    labels.to(self.device),
                )

                output = self.model(images)
                predictions = sigmoid(output)

                val_gts.extend(np.where(labels.detach().cpu().numpy() > 0.5, 1, 0))
                val_res.extend(np.where(predictions.detach().cpu().numpy() > 0.5, 1, 0))

            val_f1score = f1_score(val_gts,val_res,average='weighted')
            val_met = {"val_f1score" : val_f1score}
            log.update(**{k: v for k, v in val_met.items()})

        self.logger.info(
            "[{}/{}] Start to evaluate in the test set.".format(epoch, self.epochs)
        )

        self.model.eval()
        with torch.no_grad():
            test_gts, test_res = [], []

            for batch_idx, (
                _,
                images,
                labels,
            ) in enumerate(self.test_dataloader):
                images, labels = (
                    images.to(self.device),
                    labels.to(self.device),
                )

                output = self.model(images)
                predictions = sigmoid(output)

                test_gts.extend(np.where(labels.detach().cpu().numpy() > 0.5, 1, 0))
                test_res.extend(np.where(predictions.detach().cpu().numpy() > 0.5, 1, 0))

            test_f1score = f1_score(test_gts, test_res,average='weighted')
            test_met = {"test_f1score" : test_f1score}
            log.update(**{k: v for k, v in test_met.items()})
        wandb_dictionary.update(log)
        wandb.log(wandb_dictionary)
        self.lr_scheduler.step(val_f1score)
        return log
