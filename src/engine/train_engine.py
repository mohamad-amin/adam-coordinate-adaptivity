import math
import os
import time

import torch

from src.engine.base.base_engine import BaseEngine
from src.util.util import write_stats


class TrainEngine(BaseEngine):

    def register_configs(self):
        self.train_config = self.config['train']
        self.eval_config = self.config['eval']

    def prepare(self):
        super().prepare()
        self._prepare_optimizer(self.train_config, self.model_config)
        if self.checkpoint is not None:
            self.checkpoint_data = {k: self.checkpoint[k] for k in {'iter_num', 'best_val_loss'}}
        else:
            self.checkpoint_data = None

    def _prepare_optimizer(self, train_config, model_config):
        optimizer = self.model.configure_optimizers(train_config, self.device_type)
        if model_config['init'] == 'resume':
            optimizer.load_state_dict(self.checkpoint['optimizer'])
        self.optimizer = optimizer

    # learning rate decay scheduler (cosine with warmup)
    def get_lr(self, it):
        lr_config = self.train_config['lr']
        if not lr_config['decay']:
            return lr_config['max']
        # 1) linear warmup for warmup_iters steps
        if it < lr_config['warmup_iters']:
            return lr_config['max'] * it / lr_config['warmup_iters']
        # 2) if it > lr_decay_iters, return min learning rate
        if it > lr_config['decay_iters']:
            return lr_config['min']
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - lr_config['warmup_iters']) / (lr_config['decay_iters'] - lr_config['warmup_iters'])
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return lr_config['min'] + coeff * (lr_config['max'] - lr_config['min'])

    # helps estimate an arbitrarily accurate loss over either split using many batches
    @torch.no_grad()
    def _estimate_loss(self, get_batch):
        out = {}
        self.model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(self.eval_config['iters'])
            for k in range(self.eval_config['iters']):
                X, Y = get_batch(split)
                with self.ctx:
                    logits, loss = self.model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        self.model.train()
        return out

    # Keep track of parameters
    @torch.no_grad()
    def _measure_params(self):
        out = {}
        self.model.eval()
        if self.ddp:
            m = self.model.module
        else:
            m = self.model
        l1_norms = []
        l2_norms = []
        linf_norms = []
        for n, p in m.named_parameters():
            out['L2_' + n] = torch.norm(p)
            out['L1_' + n] = torch.norm(p, 1)
            out['Linf_' + n] = torch.norm(p, float('inf'))
            l1_norms.append(out['L1_' + n])
            l2_norms.append(out['L2_' + n] ** 2)
            linf_norms.append(out['Linf_' + n])
        out['Linf'] = torch.max(torch.tensor(linf_norms))
        out['L1'] = torch.tensor(l1_norms).sum()
        out['L2'] = torch.tensor(l1_norms).sum().sqrt()
        self.model.train()
        return out

    # Keep track of gradients
    @torch.no_grad()
    def _measure_grads(self):
        out = {}
        self.model.eval()
        if self.ddp:
            m = self.model.module
        else:
            m = self.model
        l1_norms = []
        l2_norms = []
        linf_norms = []
        for n, p in m.named_parameters():
            out['L2_' + n] = torch.norm(p.grad)
            out['L1_' + n] = torch.norm(p.grad, 1)
            out['Linf_' + n] = torch.norm(p.grad, float('inf'))
            l1_norms.append(out['L1_' + n])
            l2_norms.append(out['L2_' + n] ** 2)
            linf_norms.append(out['Linf_' + n])
        out['Linf'] = torch.max(torch.tensor(linf_norms))
        out['L1'] = torch.tensor(l1_norms).sum()
        out['L2'] = torch.tensor(l1_norms).sum().sqrt()
        self.model.train()
        return out

    def forward_backward(self, gradient_accumulation_steps, get_batch, X, Y, scaler):

        # forward backward update, with optional gradient accumulation to simulate larger batch size
        # and using the GradScaler if data type is float16
        for micro_step in range(gradient_accumulation_steps):
            if self.ddp:
                # in DDP training we only need to sync gradients at the last micro step.
                # the official way to do this is with model.no_sync() context manager, but
                # I really dislike that this bloats the code and forces us to repeat code
                # looking at the source of that context manager, it just toggles this variable
                self.model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
            with self.ctx:
                logits, loss = self.model(X, Y)
                loss = loss / gradient_accumulation_steps  # scale the loss to account for gradient accumulation
            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            X, Y = get_batch('train')
            # backward pass, with gradient scaling if training in fp16
            scaler.scale(loss).backward()

        return X, Y, loss

    def get_raw_model(self):
        return self.unwrap_ddp(self.model)

    def run(self):

        batch_size, block_size = self.train_config['batch']['size'], self.train_config['batch']['block']
        gradient_accumulation_steps = self.train_config['batch']['gradient_accumulation_steps'] // self.world_size
        get_batch = lambda split: self.get_batch(split, batch_size, block_size)

        if self.checkpoint_data is not None:
            iter_num = self.checkpoint_data['iter_num']
            best_val_loss = self.checkpoint_data['best_val_loss']
        else:
            iter_num = 0
            best_val_loss = 1e9

        scaler = torch.cuda.amp.GradScaler(enabled=(self.dtype == 'float16'))

        # training loop
        X, Y = get_batch('train')  # fetch the very first batch

        if self.train_config.get('manual_investigation', False):
            import IPython; IPython.embed()

        t0 = time.time()
        local_iter_num = 0  # number of iterations in the lifetime of this process
        running_mfu = -1.0
        while True:

            # determine and set the learning rate for this iteration
            if self.train_config['optimizer']['name'] != 'sfsgd':
                lr = self.get_lr(iter_num)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
            else:
                lr = self.optimizer.param_groups[0]['lr']

            # evaluate the loss on train/val sets and write checkpoints
            if iter_num % self.eval_config['interval'] == 0 and self.writer:
                losses = self._estimate_loss(get_batch)
                print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
                if self.writer:
                    params_stats = self._measure_params()
                    write_stats({'train': {
                        't_loss': losses['train'],
                        'e_loss': losses['val'],
                        'lr': lr,
                        "mfu": running_mfu * 100
                    }, 'weights': params_stats}, self.writer, step=iter_num)

                if losses['val'] < best_val_loss or self.log_config['always_save_checkpoint']:
                    best_val_loss = losses['val']
                    # if iter_num > 0:
                    if True: # always saving checkpoint
                        checkpoint = self.make_checkpoint()
                        checkpoint.update({
                            'iter_num': iter_num,
                            'best_val_loss': best_val_loss
                        })
                        print(f"saving checkpoint to {self.save_dir}")
                        torch.save(checkpoint, os.path.join(self.checkpoint_dir, 'ckpt.pt'))
                        if iter_num % self.log_config['save_interval'] == 0:
                            torch.save(checkpoint, os.path.join(self.checkpoint_dir, f'ckpt_{iter_num}.pt'))
                        del checkpoint

            if iter_num == 0 and not self.train_config['active']:
                break

            X, Y, loss = self.forward_backward(gradient_accumulation_steps, get_batch, X, Y, scaler)

            # recording gradient info
            if iter_num % self.eval_config['interval'] == 0 and self.writer:
                grad_stats = self._measure_grads()
                write_stats({'grad': grad_stats}, self.writer, iter_num)

            # clip the gradient
            if self.train_config.get('grad_clip', 0.0) != 0.0:
                scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.train_config['grad_clip'])
            # step the optimizer and scaler if training in fp16
            scaler.step(self.optimizer)
            scaler.update()
            # flush the gradients as soon as we can, no need for this memory anymore
            self.optimizer.zero_grad(set_to_none=True)

            # timing and logging
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            if iter_num % self.log_config['log_interval'] == 0 and self.writer:
                # get loss as float. note: this is a CPU-GPU sync point
                # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
                lossf = loss.item() * gradient_accumulation_steps
                if local_iter_num >= 5:  # let the training loop settle a bit
                    mfu = self.get_raw_model().estimate_mfu(batch_size * gradient_accumulation_steps, dt)
                    running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
                print(f"iter {iter_num}: loss {lossf:.4f}, time {dt * 1000:.2f}ms, mfu {running_mfu * 100:.2f}%")
            iter_num += 1
            local_iter_num += 1

            # termination conditions
            if iter_num > self.train_config['max_iters']:
                break

    def make_checkpoint(self):
        return {
            'model': self.get_raw_model().state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'model_args': self.model_args,
            'config': self.config,
            'wandb_id': self.id
        }
