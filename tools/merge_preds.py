import matplotlib.pyplot as plt
import numpy as np
import glob
from torch import nn
import torch 
import rasterio

from torch.optim import AdamW, SGD
from torch.cuda.amp import GradScaler, autocast

from semseg import schedulers

from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

@torch.no_grad()
def evaluate(model, input_data, labels):

    print('Evaluating...')
    model.eval()

    # j = JaccardIndex(num_classes=2)
    # d = Dice(average='micro')
    # f1 = F1Score(num_classes=2, mdmc_average='samplewise')

    # if isinstance(dataloader.dataset, torch.utils.data.Subset):
    #     metrics = Metrics(dataloader.dataset.dataset.n_classes, dataloader.dataset.dataset.ignore_label, device)
    # else:
    #     metrics = Metrics(dataloader.dataset.n_classes, dataloader.dataset.ignore_label, device)

    loss = nn.CrossEntropyLoss(ignore_index=-1)
    # loss_eval = []

    # for images, labels in tqdm(dataloader):
    #     images = images.to(device)
    #     labels = labels.to(device)
        
    preds = model(input_data)
    loss_val = loss(preds, labels)

    #metrics.update(preds.softmax(dim=1), labels)
    
    # ious, miou = metrics.compute_iou()
    # acc, macc = metrics.compute_pixel_acc()
    # f1, mf1 = metrics.compute_f1()

    #return acc, macc, f1, mf1, ious, miou, torch.mean(torch.stack(loss_eval)).detach().numpy()
    #return acc, macc, f1, mf1, ious, miou, torch.mean(torch.stack(loss_eval)).detach().cpu().numpy()

    return loss_val

    

class mergeNet(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels=4, out_channels=2, kernel_size=5, stride=1, padding='same', bias=True)
        torch.nn.init.xavier_uniform(self.conv.weight)
        
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.conv(x)
        return result

    # def init_weights(self, m):
    #     if isinstance(m, nn.Linear):
    #         torch.nn.init.xavier_uniform(m.weight)
    #         m.bias.data.fill_(0.01)


def read_sample(in_fname, out_fname):

    with rasterio.open(in_fname) as src:
        in_preds = src.read()
    
    with rasterio.open(out_fname) as src:
        out_lbl = src.read()

    out_lbl = out_lbl[2]

    # re-mapping
    out_lbl[out_lbl == 128] = 1
    out_lbl[out_lbl == 255] = 2

    out_lbl = np.expand_dims(out_lbl, 0) # []
    in_preds = np.expand_dims(in_preds, 0) # [Batch, Preds, H, W]

    out_lbl = torch.tensor(out_lbl).long() - 1
    in_preds = torch.tensor(in_preds)

    return in_preds, out_lbl


def get_optimizer(model: nn.Module, optimizer: str, lr: float = 0.001, weight_decay: float = 0.01):
    
    wd_params, nwd_params = [], []
    for p in model.parameters():
        if p.dim() == 1:
            nwd_params.append(p)
        else:
            wd_params.append(p)
    
    params = \
    [
        {"params": wd_params},
        {"params": nwd_params, "weight_decay": 0}
    ]

    if optimizer == 'adamw':
        return AdamW(params, lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=weight_decay)
    else:
        return SGD(params, lr, momentum=0.9, weight_decay=weight_decay)



class OhemCrossEntropy(nn.Module):
    
    def __init__(self, ignore_label: int = 255, weight: torch.Tensor = None, thresh: float = 0.7, aux_weights: list = [1, 1]) -> None:
        
        super().__init__()
        self.ignore_label = ignore_label
        self.aux_weights = aux_weights
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float))
        self.criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_label, reduction='none')

    def _forward(self, preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        
        # preds in shape [B, C, H, W] and labels in shape [B, H, W]
        n_min = labels[labels != self.ignore_label].numel() // 16
        loss = self.criterion(preds, labels).view(-1)
        loss_hard = loss[loss > self.thresh]

        if loss_hard.numel() < n_min:
            loss_hard, _ = loss.topk(n_min)

        return torch.mean(loss_hard)

    def forward(self, preds, labels: torch.Tensor) -> torch.Tensor:
        
        if isinstance(preds, tuple):
            return sum([w * self._forward(pred, labels) for (pred, w) in zip(preds, self.aux_weights)])
        return self._forward(preds, labels)



if __name__ == '__main__':

    train_inputs = '/censipam_data/renam/netMerge/output/cpred*.tif'
    train_labels = '/censipam_data/renam/netMerge/input/d*/masked.tif'

    train_inputs = glob.glob(train_inputs)
    train_labels = glob.glob(train_labels)

    train_inputs.sort()
    train_labels.sort()


    val_inputs = train_inputs[0]
    val_labels = train_labels[0]

    train_inputs = train_inputs[1:]
    train_labels = train_labels[1:]

    nb_epochs = 50
    batch_size = 1
    iters_per_epoch = len(train_inputs) // batch_size

    model = mergeNet()


    optimizer = get_optimizer(model, 'adamw')
    scheduler = schedulers.get_scheduler('warmuppolylr', optimizer, \
        max_iter = nb_epochs * iters_per_epoch, \
        power = 0.9, \
        warmup_iter=  2, \
        warmup_ratio= 0.1)

    loss_fn = OhemCrossEntropy(ignore_label = -1)
    scaler = GradScaler(enabled=False)

    train_loss = []
    writer = SummaryWriter( Path('/censipam_data/renam/netMerge') / 'logs' )

    min_val_loss = 1e9

    for epoch in range(nb_epochs):

        model.train()
        train_loss = 0.0

        # train
        for i in range(len(train_inputs)):
            
            input_preds, lbl = read_sample(train_inputs[i], train_labels[i])
            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=False):
                logits = model(input_preds)        
                loss = loss_fn(logits, lbl)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
                
            lr = scheduler.get_lr()
            lr = sum(lr) / len(lr)
            train_loss += loss.item()

            print(f'Train loss: {loss.item()}')

            writer.add_scalars('Loss', {'train': loss}, epoch)

        train_loss /= len(train_inputs)

        # validation
        input_preds, lbl = read_sample(val_inputs, val_labels)

        val_loss = evaluate(model, input_preds, lbl)

        print(f'Val loss: {val_loss.item()}')
        writer.add_scalars('Loss', {'val': val_loss}, epoch)

        if val_loss < min_val_loss:
            
            min_val_loss = val_loss
        
            #torch.save(model.module.state_dict() if train_cfg['DDP'] else model.state_dict(), save_dir / f"{model_cfg['NAME']}_{model_cfg['BACKBONE']}_{dataset_cfg['NAME']}.pth")

            torch.save(model.state_dict(), Path('/censipam_data/renam/netMerge') / 'model_merge.pth' )

        

        print('\n\n')
        print(train_loss)