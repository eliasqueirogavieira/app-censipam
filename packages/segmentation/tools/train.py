from cProfile import label
import torch 
import argparse
import yaml
import time
import multiprocessing as mp
from tabulate import tabulate
from tqdm import tqdm
from torch.utils.data import DataLoader
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler, RandomSampler
from torch import distributed as dist
from semseg.models import *
from semseg.datasets import * 
from semseg.augmentations import get_train_augmentation, get_val_augmentation
from semseg.losses import get_loss
from semseg.schedulers import get_scheduler
from semseg.optimizers import get_optimizer
from semseg.utils.utils import fix_seeds, setup_cudnn, cleanup_ddp, setup_ddp
from val import evaluate

from torchvision.utils import make_grid

from censipam import Censipam
import censipam as cspm


def main(cfg, gpu, save_dir):

    start = time.time()
    best_mIoU = 0.0
    num_workers = mp.cpu_count()
    device = torch.device(cfg['DEVICE'])
    train_cfg, eval_cfg = cfg['TRAIN'], cfg['EVAL']
    dataset_cfg, model_cfg = cfg['DATASET'], cfg['MODEL']
    loss_cfg, optim_cfg, sched_cfg = cfg['LOSS'], cfg['OPTIMIZER'], cfg['SCHEDULER']
    epochs, lr = train_cfg['EPOCHS'], optim_cfg['LR']
    
    traintransform = cspm.get_train_augmentation(train_cfg['IMAGE_SIZE'], seg_fill=0)
    #valtransform = get_val_augmentation(eval_cfg['IMAGE_SIZE'])

    trainset = eval(dataset_cfg['NAME'])(dataset_cfg['ROOT'], 'train', ignore_lbl=dataset_cfg['IGNORE_LABEL'], transform=traintransform)

    nb_val = trainset.__len__() * 10 // 100
    trainsubset, valsubset = torch.utils.data.random_split(trainset, [trainset.__len__()-nb_val, nb_val])
    
    model = eval(model_cfg['NAME'])(model_cfg['BACKBONE'], trainset.n_classes)
    model.init_pretrained(model_cfg['PRETRAINED'])
    model = model.to(device)
    print("{} - #parameters: {} ".format( model_cfg['NAME'], sum([p.numel() for p in model.parameters()]))) 

    if train_cfg['DDP']: 
        sampler = DistributedSampler(trainset, dist.get_world_size(), dist.get_rank(), shuffle=True)
        model = DDP(model, device_ids=[gpu])
    else:
        #sampler = RandomSampler(trainset)
        sampler = RandomSampler(trainsubset)
    
    #trainloader = DataLoader(trainset, batch_size=train_cfg['BATCH_SIZE'], num_workers=num_workers, drop_last=True, pin_memory=True, sampler=sampler)
    #valloader = DataLoader(valset, batch_size=1, num_workers=1, pin_memory=True)

    trainloader = DataLoader(trainsubset, batch_size=train_cfg['BATCH_SIZE'], num_workers=num_workers, drop_last=True, pin_memory=True, sampler=sampler)
    valloader = DataLoader(valsubset, batch_size=1, num_workers=1, pin_memory=True)

    #iters_per_epoch = len(trainset) // train_cfg['BATCH_SIZE']
    iters_per_epoch = len(trainsubset) // train_cfg['BATCH_SIZE']

    # class_weights = trainset.class_weights.to(device)
    loss_fn = get_loss(loss_cfg['NAME'], trainset.ignore_label, None)
    optimizer = get_optimizer(model, optim_cfg['NAME'], lr, optim_cfg['WEIGHT_DECAY'])
    scheduler = get_scheduler(sched_cfg['NAME'], optimizer, epochs * iters_per_epoch, sched_cfg['POWER'], iters_per_epoch * sched_cfg['WARMUP'], sched_cfg['WARMUP_RATIO'])
    scaler = GradScaler(enabled=train_cfg['AMP'])
    writer = SummaryWriter(str(save_dir / 'logs'))

    for epoch in range(epochs):
        model.train()
        if train_cfg['DDP']: sampler.set_epoch(epoch)

        train_loss = 0.0
        pbar = tqdm(enumerate(trainloader), total=iters_per_epoch, desc=f"Epoch: [{epoch+1}/{epochs}] Iter: [{0}/{iters_per_epoch}] LR: {lr:.8f} Loss: {train_loss:.8f}")

        for iter, (img, lbl) in pbar:
            optimizer.zero_grad(set_to_none=True)

            img = img.to(device)
            lbl = lbl.to(device)
            
            with autocast(enabled=train_cfg['AMP']):
                logits = model(img)
                scores = logits.softmax(1) # only for SFnet
                loss = loss_fn(logits, lbl)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            if train_cfg['DDP']: 
                torch.cuda.synchronize()

            lr = scheduler.get_lr()
            lr = sum(lr) / len(lr)
            train_loss += loss.item()

            pbar.set_description(f"Epoch: [{epoch+1}/{epochs}] Iter: [{iter+1}/{iters_per_epoch}] LR: {lr:.8f} Loss: {train_loss / (iter+1):.8f}")
        
        train_loss /= iter+1
        writer.add_scalars('Loss', {'train': loss}, epoch)
        torch.cuda.empty_cache()

        if (epoch+1) % train_cfg['EVAL_INTERVAL'] == 0 or (epoch+1) == epochs:
            
            acc, macc, f1, mf1, ious, miou, loss = evaluate(model, valloader, device)

            lbl_tmp = torch.unsqueeze(lbl, 1)

            writer.add_scalars('Loss', {'val': loss}, epoch)
            writer.add_scalars('Metrics/mIoU', {'val': miou}, epoch)
            writer.add_scalars('Metrics/f1', {'val': mf1}, epoch)
            
            img_grid = make_grid( img[:min(cfg['TRAIN']['BATCH_SIZE'], 4), 0:1, ...] )
            writer.add_image('Image/VV', img_grid, epoch)

            img_grid = make_grid( img[:min(cfg['TRAIN']['BATCH_SIZE'], 4), 1:2, ...] )
            writer.add_image('Image/VH', img_grid, epoch)

            img_grid = make_grid( lbl_tmp[:min(cfg['TRAIN']['BATCH_SIZE'], 4), 0:1, ...] )
            writer.add_image('Images/GT', img_grid, epoch)

            img_grid = make_grid( scores[:min(cfg['TRAIN']['BATCH_SIZE'], 4), 1:2, ...] )
            writer.add_image('Prediction', img_grid, epoch)

            if miou > best_mIoU:
                best_mIoU = miou
                torch.save(model.module.state_dict() if train_cfg['DDP'] else model.state_dict(), save_dir / f"{model_cfg['NAME']}_{model_cfg['BACKBONE']}_{dataset_cfg['NAME']}.pth")
            print(f"Current mIoU: {miou} Best mIoU: {best_mIoU}")

    writer.close()
    pbar.close()
    end = time.gmtime(time.time() - start)

    table = [
        ['Best mIoU', f"{best_mIoU:.2f}"],
        ['Total Training Time', time.strftime("%H:%M:%S", end)]
    ]
    print(tabulate(table, numalign='right'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/custom.yaml', help='Configuration file to use')
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    fix_seeds(3407)
    setup_cudnn()
    gpu = setup_ddp()
    save_dir = Path(cfg['SAVE_DIR'])
    save_dir.mkdir(exist_ok=True)
    main(cfg, gpu, save_dir)
    cleanup_ddp()