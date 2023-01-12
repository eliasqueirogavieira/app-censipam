import torch
from torch import Tensor
from typing import Tuple

from torchmetrics import F1Score, Dice, JaccardIndex
from torchmetrics.functional import precision_recall

# precision_recall(pred, target,  mdmc_average='samplewise' )
# precision_recall(pred, target,  mdmc_average='global' )

class Metrics:
    def __init__(self, num_classes: int, ignore_label: int, device) -> None:
        self.ignore_label = ignore_label
        self.num_classes = num_classes
        self.hist = torch.zeros(num_classes, num_classes).to(device)

    def update(self, pred: Tensor, target: Tensor) -> None:
        pred = pred.argmax(dim=1)
        keep = target != self.ignore_label
        self.hist += torch.bincount(target[keep] * self.num_classes + pred[keep], minlength=self.num_classes**2).view(self.num_classes, self.num_classes)

        # prec_recall = precision_recall(pred, target, num_classes=2, average='macro',  mdmc_average='samplewise')
        # f1 = F1Score(num_classes=2, average='macro', mdmc_average='samplewise')(pred, target)
        # dice = Dice(num_classes = 2, average='macro')(pred, target)
        # jaccard = JaccardIndex(num_classes=2)(pred, target)

        # self.torchMetrics = {'f1': f1, 'dice': dice, 'jaccard': jaccard, 'prec_recall': prec_recall}


    def compute_iou(self) -> Tuple[Tensor, Tensor]:
        ious = self.hist.diag() / (self.hist.sum(0) + self.hist.sum(1) - self.hist.diag())
        miou = ious[~ious.isnan()].mean().item()
        ious *= 100
        miou *= 100
        return ious.cpu().numpy().round(2).tolist(), round(miou, 2)

    def compute_f1(self) -> Tuple[Tensor, Tensor]:
        f1 = 2 * self.hist.diag() / (self.hist.sum(0) + self.hist.sum(1))
        mf1 = f1[~f1.isnan()].mean().item()
        f1 *= 100
        mf1 *= 100
        return f1.cpu().numpy().round(2).tolist(), round(mf1, 2)

    def compute_pixel_acc(self) -> Tuple[Tensor, Tensor]:
        acc = self.hist.diag() / self.hist.sum(1)
        macc = acc[~acc.isnan()].mean().item()
        acc *= 100
        macc *= 100
        return acc.cpu().numpy().round(2).tolist(), round(macc, 2)

