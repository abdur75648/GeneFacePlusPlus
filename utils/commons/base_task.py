import logging
import os
import time
import random
import subprocess
import sys
from datetime import datetime
import numpy as np
import torch.utils.data
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from utils.commons.dataset_utils import data_loader
from utils.commons.hparams import hparams
from utils.commons.meters import AvgrageMeter
from utils.commons.tensor_utils import tensors_to_scalars
from utils.commons.trainer import Trainer
from utils.nn.model_utils import print_arch, num_params

torch.multiprocessing.set_sharing_strategy(os.getenv('TORCH_SHARE_STRATEGY', 'file_system'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')


class BaseTask(nn.Module):
    def __init__(self, *args, **kwargs):
        super(BaseTask, self).__init__()
        self.current_epoch = 0
        self.global_step = 0
        self.trainer = None
        self.use_ddp = False
        self.gradient_clip_norm = hparams['clip_grad_norm']
        self.gradient_clip_val = hparams.get('clip_grad_value', 0)
        self.model = None
        self.epoch_training_losses_meter = None
        self.logger: SummaryWriter = None

    ######################
    # build model, dataloaders, optimizer, scheduler and tensorboard
    ######################
    def build_model(self):
        raise NotImplementedError

    @data_loader
    def train_dataloader(self):
        raise NotImplementedError

    @data_loader
    def test_dataloader(self):
        raise NotImplementedError

    @data_loader
    def val_dataloader(self):
        raise NotImplementedError

    def build_scheduler(self, optimizer):
        return None

    def build_optimizer(self, model):
        raise NotImplementedError

    def configure_optimizers(self):
        optm = self.build_optimizer(self.model)
        self.scheduler = self.build_scheduler(optm)
        if isinstance(optm, (list, tuple)):
            return optm
        return [optm]

    def build_tensorboard(self, save_dir, name, **kwargs):
        log_dir = os.path.join(save_dir, name)
        os.makedirs(log_dir, exist_ok=True)
        self.logger = SummaryWriter(log_dir=log_dir, **kwargs)

    ######################
    # training
    ######################
    def on_train_start(self):
        for n, m in self.model.named_children():
            num_params(m, model_name=n)
        if torch.__version__.split(".")[0] == '2' and hparams.get("torch_compile", False):
            self.model = torch.compile(self.model, mode='default')

    def on_train_end(self):
        pass

    def on_epoch_start(self):
        self.epoch_training_losses_meter = {'total_loss': AvgrageMeter()}

    def on_epoch_end(self):
        loss_outputs = {k: v.avg for k, v in self.epoch_training_losses_meter.items()}
        print(f"Epoch {self.current_epoch} ended. Steps: {self.global_step}. {loss_outputs}")
        loss_outputs = {"epoch_mean/"+k:v for k,v in loss_outputs.items()}
        return loss_outputs

    def _training_step(self, sample, batch_idx, optimizer_idx):
        """

        :param sample:
        :param batch_idx:
        :return: total loss: torch.Tensor, loss_log: dict
        """
        raise NotImplementedError

    def training_step(self, sample, batch_idx, optimizer_idx=-1):
        """

        :param sample:
        :param batch_idx:
        :param optimizer_idx:
        :return: {'loss': torch.Tensor, 'progress_bar': dict, 'tb_log': dict}
        """
        # perform the main training step in a specific task
        loss_ret = self._training_step(sample, batch_idx, optimizer_idx)
        if loss_ret is None:
            return {'loss': None}
        total_loss, log_outputs = loss_ret
        log_outputs = tensors_to_scalars(log_outputs)

        # add to epoch meter
        for k, v in log_outputs.items():
            if '/' in k:
                k_split = k.split("/")
                assert len(k_split) == 2, "we only support one `/` in tag_name, i.e., `<tag>/<sub_tag>`"
                k = k.replace("/", "_")
            if k not in self.epoch_training_losses_meter:
                self.epoch_training_losses_meter[k] = AvgrageMeter()
            if not np.isnan(v):
                self.epoch_training_losses_meter[k].update(v)
        
        if optimizer_idx >= 0:
            for params_group_i in range(len(self.trainer.optimizers[optimizer_idx].param_groups)):
                log_outputs[f'lr/optimizer{optimizer_idx}_params_group{params_group_i}'] = self.trainer.optimizers[optimizer_idx].param_groups[params_group_i]['lr']

        # add to progress bar
        progress_bar_log = {}
        for k, v in log_outputs.items():
            if '/' in k:
                k_split = k.split("/")
                assert len(k_split) == 2, "we only support one `/` in tag_name, i.e., `<tag>/<sub_tag>`"
                k = k.replace("/", "_")
            assert k not in progress_bar_log, f"we got duplicate tags in log_outputs, check this `{k}`"
            progress_bar_log[k] = v

        # add to progress bar
        tb_log = {}
        for k, v in log_outputs.items():
            if '/' in k:
                tb_log[k] = v
            else:
                tb_log[f'tr/{k}'] = v

        if not isinstance(total_loss, torch.Tensor):
            return {'loss': None}
        self.epoch_training_losses_meter['total_loss'].update(total_loss.item())

        return {
            'loss': total_loss,
            'progress_bar': progress_bar_log,
            'tb_log': tb_log
        }

    def on_before_optimization(self, opt_idx):
        if self.gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.gradient_clip_norm)
        if self.gradient_clip_val > 0:
            torch.nn.utils.clip_grad_value_(self.parameters(), self.gradient_clip_val)

    def on_after_optimization(self, epoch, batch_idx, optimizer, optimizer_idx):
        if self.scheduler is not None:
            self.scheduler.step(self.global_step // hparams['accumulate_grad_batches'])

    ######################
    # validation
    ######################
    def validation_start(self):
        pass

    def validation_step(self, sample, batch_idx):
        """

        :param sample:
        :param batch_idx:
        :return: output: {"losses": {...}, "total_loss": float, ...} or (total loss: torch.Tensor, loss_log: dict)
        """
        raise NotImplementedError

    def validation_end(self, outputs):
        """

        :param outputs:
        :return: loss_output: dict
        """
        all_losses_meter = {'total_loss': AvgrageMeter()}
        for output in outputs:
            if output is None or len(output) == 0:
                continue
            if isinstance(output, dict):
                assert 'losses' in output, 'Key "losses" should exist in validation output.'
                n = output.pop('nsamples', 1)
                losses = tensors_to_scalars(output['losses'])
                total_loss = output.get('total_loss', sum(losses.values()))
            else:
                assert len(output) == 2, 'Validation output should only consist of two elements: (total_loss, losses)'
                n = 1
                total_loss, losses = output
                losses = tensors_to_scalars(losses)
            if isinstance(total_loss, torch.Tensor):
                total_loss = total_loss.item()
            for k, v in losses.items():
                if k not in all_losses_meter:
                    all_losses_meter[k] = AvgrageMeter()
                all_losses_meter[k].update(v, n)
            all_losses_meter['total_loss'].update(total_loss, n)
        loss_output = {k: round(v.avg, 10) for k, v in all_losses_meter.items()}
        print(f"| Validation results@{self.global_step}: {loss_output}")
        return {
            'tb_log': {f'val/{k}': v for k, v in loss_output.items()},
            'val_loss': loss_output['total_loss']
        }

    ######################
    # testing
    ######################
    def test_start(self):
        pass

    def test_step(self, sample, batch_idx):
        return self.validation_step(sample, batch_idx)

    def test_end(self, outputs):
        return self.validation_end(outputs)

    ######################
    # start training/testing
    ######################
    @classmethod
    def start(cls):

        def is_port_in_use(port: int) -> bool:
            import socket
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                return s.connect_ex(('localhost', port)) == 0

        os.environ['MASTER_PORT'] = str(random.randint(10000, 11000))
        while is_port_in_use(int(os.environ['MASTER_PORT'])):
            print(f"| Port {os.environ['MASTER_PORT']} is in use. Change another port...")
            os.environ['MASTER_PORT'] = str(random.randint(10000, 11000))
            time.sleep(1)

        random.seed(hparams['seed'])
        np.random.seed(hparams['seed'])
        work_dir = hparams['work_dir']
        trainer = Trainer(
            work_dir=work_dir,
            val_check_interval=hparams['val_check_interval'],
            tb_log_interval=hparams['tb_log_interval'],
            max_updates=hparams['max_updates'],
            num_sanity_val_steps=hparams['num_sanity_val_steps'] if not hparams['validate'] else 10000,
            accumulate_grad_batches=hparams['accumulate_grad_batches'],
            print_nan_grads=hparams['print_nan_grads'],
            resume_from_checkpoint=hparams.get('resume_from_checkpoint', 0),
            amp=hparams['amp'],
            monitor_key=hparams['valid_monitor_key'],
            monitor_mode=hparams['valid_monitor_mode'],
            num_ckpt_keep=hparams['num_ckpt_keep'],
            save_best=hparams['save_best'],
            seed=hparams['seed'],
            debug=hparams['debug']
        )
        if not hparams['infer']:  # train
            trainer.fit(cls)
        else:
            trainer.test(cls)

    def on_keyboard_interrupt(self):
        pass
