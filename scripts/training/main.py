# Modifications for PCFM © 2025 Pengfei Cai (Learning Matter @ MIT) and Utkarsh (Julia Lab @ MIT), licensed under the MIT License.
# Original portions © Amazon.com, Inc. or its affiliates, licensed under the Apache License 2.0.

import argparse
import os
import time

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import get_dataset
from models import get_flow_model
from scripts.training.logger import ExperimentLogger, build_run_metadata
from scripts.training.utils import (
    count_parameters,
    get_optimizer,
    get_scheduler,
    load_config,
    seed_all,
)
from scripts.training.vis_utils import draw

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--mode', type=str, choices=['train', 'inf'], default='train')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--logdir', type=str, default='./logs')
    parser.add_argument('--savename', type=str, default='test')
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    seed_all(config.train.seed)
    print(config)
    logdir = os.path.join(args.logdir, args.savename)
    os.makedirs(logdir, exist_ok=True)

    exp_logger = ExperimentLogger(config, config_path=args.config)

    print('Loading datasets...')
    train_set, test_set = get_dataset(config.datasets)

    train_loader = DataLoader(train_set, batch_size=config.train.batch_size, shuffle=True, num_workers=16)
    test_loader  = DataLoader(test_set,  batch_size=config.train.batch_size, shuffle=True, num_workers=8)

    print('Building model...')
    model = get_flow_model(config.model, config.encoder).to(args.device)
    print(f'Number of parameters: {count_parameters(model)}')

    optimizer = get_optimizer(config.train.optimizer, model)
    scheduler = get_scheduler(config.train.scheduler, optimizer)
    optimizer.zero_grad()

    # Log all run metadata to W&B config (no-op if W&B is disabled).
    exp_logger.log_config(build_run_metadata(config, train_set, test_set, model))

    if args.resume is not None:
        print(f'Resuming from checkpoint: {args.resume}')
        ckpt = torch.load(args.resume, map_location=args.device)
        model.load_state_dict(ckpt['model'])
        if 'optimizer' in ckpt:
            print('Resuming optimizer states...')
            optimizer.load_state_dict(ckpt['optimizer'])
        if 'scheduler' in ckpt:
            print('Resuming scheduler states...')
            scheduler.load_state_dict(ckpt['scheduler'])
        torch.cuda.empty_cache()
    global_step = 0


    def train():
        global global_step

        epoch = 0
        while True:
            model.train()
            epoch_losses = []
            for x in train_loader:
                x = x.to(args.device)
                _t0 = time.time()
                loss = model.get_loss(x)
                epoch_losses.append(loss.item())
                loss.backward()
                grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                _step_dt = time.time() - _t0

                # --- logging ---
                _metrics = {
                    'train/loss':      loss.item(),
                    'train/grad':      grad_norm.item(),
                    'train/lr':        optimizer.param_groups[0]['lr'],
                    'train/epoch':     epoch,
                    'time/step_seconds': _step_dt,
                }
                if torch.cuda.is_available():
                    _metrics['gpu/memory_allocated_gb'] = torch.cuda.memory_allocated() / 1e9
                    _metrics['gpu/memory_reserved_gb']  = torch.cuda.memory_reserved()  / 1e9
                exp_logger.log_metrics(_metrics, global_step)

                if global_step % config.train.log_freq == 0:
                    print(f'Epoch {epoch} Step {global_step} train loss {loss.item():.6f}')
                global_step += 1

                if global_step % config.train.val_freq == 0:
                    avg_val_loss = validate()
                    sample_uncond()
                    if config.train.scheduler.type == 'plateau':
                        scheduler.step(avg_val_loss)
                    else:
                        scheduler.step()

                    model.train()
                    latest_ckpt = os.path.join(logdir, 'latest.pt')
                    torch.save({
                        'model': model.state_dict(),
                        'step':  global_step,
                    }, latest_ckpt)
                    if global_step % config.train.save_freq == 0:
                        ckpt_path = os.path.join(logdir, f'{global_step}.pt')
                        torch.save({
                            'config':        config,
                            'model':         model.state_dict(),
                            'optimizer':     optimizer.state_dict(),
                            'scheduler':     scheduler.state_dict(),
                            'avg_val_loss':  avg_val_loss,
                        }, ckpt_path)
                        exp_logger.log_summary('latest_checkpoint', ckpt_path)

                if global_step >= config.train.max_iter:
                    return

            epoch_loss = sum(epoch_losses) / len(epoch_losses)
            print(f'Epoch {epoch} train loss {epoch_loss:.6f}')
            epoch += 1


    def validate():
        with torch.no_grad():
            model.eval()

            val_losses = []
            total = config.train.valid_max_batch or len(test_loader)
            total = min(total, len(test_loader))
            for i, x in tqdm(enumerate(test_loader), total=total):
                if i >= total:
                    break
                x = x.to(args.device)
                loss = model.get_loss(x)
                val_losses.append(loss.item())
        val_loss = sum(val_losses) / len(val_losses)
        exp_logger.log_metrics({'val/loss': val_loss}, global_step)
        print(f'Step {global_step} valid loss {val_loss:.6f}')
        return val_loss


    @torch.no_grad()
    def sample_uncond():
        model.eval()
        gen = model.sample(config.n_sample, config.n_eval, config.sample_dims, args.device)
        for i in range(config.n_sample):
            exp_logger.log_image(f'sample/{i}', draw(gen[i], **config.vis), global_step)
        return gen


    try:
        if args.mode == 'train':
            train()
            print('Training finished!')
            sample_uncond()
            print('Sampling finished!')
            final_ckpt = os.path.join(logdir, 'latest.pt')
            exp_logger.log_summary('final_checkpoint', final_ckpt)
            exp_logger.finish(final_ckpt_path=final_ckpt)
        else:
            if args.resume is None:
                print('[WARNING]: inference mode without loading a pretrained model')
            sample_uncond()
            print('Sampling finished!')
            exp_logger.finish()
    except KeyboardInterrupt:
        print('Terminating...')
        exp_logger.finish()
