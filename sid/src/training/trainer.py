import heapq
import logging
import os
from time import time

import numpy as np
import torch
from torch import optim
from transformers import get_constant_schedule_with_warmup, get_linear_schedule_with_warmup


class Trainer:
    """Minimal trainer ported from the original sid/trainer.py.

    Contract for model:
    - forward(x, ...) -> (out, quant_loss, indices)
    - compute_loss(out, quant_loss, xs=...) -> (loss_total, loss_recon)
    - get_indices(x, ...) -> indices
    """

    def __init__(self, args, model, steps_per_epoch: int):
        self.args = args
        self.model = model
        self.logger = logging.getLogger(__name__)

        self.lr = args.lr
        self.learner = args.learner
        self.lr_scheduler_type = args.lr_scheduler_type
        self.weight_decay = args.weight_decay
        self.epochs = args.epochs
        self.warmup_steps = args.warmup_epochs * steps_per_epoch
        self.max_steps = args.epochs * steps_per_epoch

        self.save_limit = args.save_limit
        self.best_save_heap = []
        self.newest_save_queue = []
        self.eval_step = min(args.eval_step, self.epochs)

        self.device = torch.device(args.device)
        self.max_batches = getattr(args, "max_batches", None)

        self.ckpt_dir = args.ckpt_dir
        os.makedirs(self.ckpt_dir, exist_ok=True)

        self.best_loss = np.inf
        self.best_collision_rate = np.inf
        self.best_loss_ckpt = "best_loss_model.pth"
        self.best_collision_ckpt = "best_collision_model.pth"

        self.optimizer = self._build_optimizer()
        self.scheduler = self._get_scheduler()

        self.model = self.model.to(self.device)

    def _build_optimizer(self):
        params = self.model.parameters()
        learner = str(self.learner).lower()

        if learner == "adam":
            return optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)
        if learner == "sgd":
            return optim.SGD(params, lr=self.lr, weight_decay=self.weight_decay)
        if learner == "adagrad":
            opt = optim.Adagrad(params, lr=self.lr, weight_decay=self.weight_decay)
            for state in opt.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(self.device)
            return opt
        if learner == "rmsprop":
            return optim.RMSprop(params, lr=self.lr, weight_decay=self.weight_decay)
        if learner == "adamw":
            return optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)

        self.logger.warning("Unknown optimizer %s, defaulting to Adam", self.learner)
        return optim.Adam(params, lr=self.lr)

    def _get_scheduler(self):
        if str(self.lr_scheduler_type).lower() == "linear":
            return get_linear_schedule_with_warmup(
                optimizer=self.optimizer,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=self.max_steps,
            )
        return get_constant_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=self.warmup_steps,
        )

    @staticmethod
    def _check_nan(loss: torch.Tensor):
        if torch.isnan(loss):
            raise ValueError("Training loss is nan")

    @torch.no_grad()
    def _post_step_hook(self):
        # Optional hook for modules that implement post-optimizer updates (e.g. EMA codebook smoothing).
        for m in self.model.modules():
            fn = getattr(m, "apply_post_ema", None)
            if callable(fn):
                fn()

    def _train_epoch(self, train_data, epoch_idx: int):
        self.model.train()
        total_loss = 0.0
        total_recon_loss = 0.0

        for batch_idx, data in enumerate(train_data):
            if self.max_batches is not None and batch_idx >= self.max_batches:
                break
            data = data.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)

            out, rq_loss, _indices = self.model(data)
            loss, loss_recon = self.model.compute_loss(out, rq_loss, xs=data)

            self._check_nan(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self._post_step_hook()
            self.scheduler.step()

            total_loss += float(loss.item())
            total_recon_loss += float(loss_recon.item())

        return total_loss, total_recon_loss

    @torch.no_grad()
    def _valid_epoch(self, valid_data):
        self.model.eval()

        indices_set = set()
        num_sample = 0
        for batch_idx, data in enumerate(valid_data):
            if self.max_batches is not None and batch_idx >= self.max_batches:
                break
            num_sample += len(data)
            data = data.to(self.device)
            indices = self.model.get_indices(data)
            indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
            for index in indices:
                indices_set.add("-".join(str(int(v)) for v in index))

        if num_sample == 0:
            return 1.0
        return (num_sample - len(indices_set)) / num_sample

    def _save_checkpoint(self, epoch: int, collision_rate: float = 1.0, ckpt_file: str | None = None):
        ckpt_path = (
            os.path.join(self.ckpt_dir, ckpt_file)
            if ckpt_file
            else os.path.join(self.ckpt_dir, f"epoch_{epoch}_collision_{collision_rate:.4f}_model.pth")
        )

        state = {
            "args_dict": getattr(self.args, "__dict__", {}),
            "epoch": epoch,
            "best_loss": self.best_loss,
            "best_collision_rate": self.best_collision_rate,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(state, ckpt_path, pickle_protocol=4)
        self.logger.info("Saving checkpoint: %s", ckpt_path)
        return ckpt_path

    def fit(self, data_loader):
        for epoch_idx in range(self.epochs):
            s_time = time()
            train_loss, train_recon_loss = self._train_epoch(data_loader, epoch_idx)
            e_time = time()
            self.logger.info(
                "epoch %d training [time: %.2fs, train loss: %.4f, recon loss: %.4f]",
                epoch_idx,
                e_time - s_time,
                train_loss,
                train_recon_loss,
            )

            if (epoch_idx + 1) % self.eval_step == 0:
                v_s = time()
                collision_rate = self._valid_epoch(data_loader)

                if train_loss < self.best_loss:
                    self.best_loss = train_loss
                    self._save_checkpoint(epoch=epoch_idx, ckpt_file=self.best_loss_ckpt)

                if collision_rate < self.best_collision_rate:
                    self.best_collision_rate = collision_rate
                    self._save_checkpoint(epoch=epoch_idx, collision_rate=collision_rate, ckpt_file=self.best_collision_ckpt)

                v_e = time()
                self.logger.info(
                    "epoch %d evaluating [time: %.2fs, collision_rate: %.6f]",
                    epoch_idx,
                    v_e - v_s,
                    collision_rate,
                )

                ckpt_path = self._save_checkpoint(epoch_idx, collision_rate=collision_rate)

                now_save = (-collision_rate, ckpt_path)
                if len(self.newest_save_queue) < self.save_limit:
                    self.newest_save_queue.append(now_save)
                    heapq.heappush(self.best_save_heap, now_save)
                else:
                    old_save = self.newest_save_queue.pop(0)
                    self.newest_save_queue.append(now_save)

                    if collision_rate < -self.best_save_heap[0][0]:
                        bad_save = heapq.heappop(self.best_save_heap)
                        heapq.heappush(self.best_save_heap, now_save)
                        if bad_save not in self.newest_save_queue:
                            try:
                                os.remove(bad_save[1])
                            except FileNotFoundError:
                                pass

                    if old_save not in self.best_save_heap:
                        try:
                            os.remove(old_save[1])
                        except FileNotFoundError:
                            pass

        return self.best_loss, self.best_collision_rate
