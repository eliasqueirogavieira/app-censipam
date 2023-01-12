import torch
import torch.nn as nn
import cv2
import numpy as np

from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import postprocess, invert_affine, display, display_annotations, undo_packing


def calc_iou(a, b):
    # a(anchor) [boxes, (y1, x1, y2, x2)]
    # b(gt, coco-style) [boxes, (x1, y1, x2, y2)]

    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    #                               x2 - y2
    iw = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 0]) # intersection width
    ih = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 1]) # intersection height
    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)
    ua = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1) + area - iw * ih
    ua = torch.clamp(ua, min=1e-8)
    intersection = iw * ih
    IoU = intersection / ua

    return IoU


class FocalLoss(nn.Module):
    def __init__(self, params):
        super(FocalLoss, self).__init__()
        self.params = params

    def forward(self, classifications, regressions, anchors, annotations, **kwargs):
        alpha = 0.25
        gamma = 2.0
        batch_size = classifications.shape[0]
        classification_losses = []
        regression_losses = []

        anchor = anchors[0, :, :]  # assuming all image sizes are the same, which it is
        dtype = anchors.dtype

        anchor_widths = anchor[:, 3] - anchor[:, 1] # xmax - x_min (largura) or col_max - col_min
        anchor_heights = anchor[:, 2] - anchor[:, 0]
        anchor_ctr_x = anchor[:, 1] + 0.5 * anchor_widths
        anchor_ctr_y = anchor[:, 0] + 0.5 * anchor_heights

        for j in range(batch_size): # for each image

            classification = classifications[j, :, :]
            regression = regressions[j, :, :]

            bbox_annotation = annotations[j]
            bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1] # bbox having a class 

            classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4) # clip to min and max values
            
            # if there is no annotation
            if bbox_annotation.shape[0] == 0:
                if torch.cuda.is_available():
                    
                    alpha_factor = torch.ones_like(classification) * alpha
                    alpha_factor = alpha_factor.cuda()
                    alpha_factor = 1. - alpha_factor
                    focal_weight = classification
                    focal_weight = alpha_factor * torch.pow(focal_weight, gamma)
                    
                    bce = -(torch.log(1.0 - classification))
                    cls_loss = focal_weight * bce

                    regression_losses.append(torch.tensor(0).to(dtype).cuda())
                    classification_losses.append(cls_loss.sum())
                else:
                    
                    alpha_factor = torch.ones_like(classification) * alpha
                    alpha_factor = 1. - alpha_factor
                    focal_weight = classification
                    focal_weight = alpha_factor * torch.pow(focal_weight, gamma)
                    
                    bce = -(torch.log(1.0 - classification))
                    
                    cls_loss = focal_weight * bce
                    
                    regression_losses.append(torch.tensor(0).to(dtype))
                    classification_losses.append(cls_loss.sum())

                continue
                
            IoU = calc_iou(anchor[:, :], bbox_annotation[:, :4])

            IoU_max, IoU_argmax = torch.max(IoU, dim=1)

            # compute the loss for classification
            targets = torch.ones_like(classification) * -1
            if torch.cuda.is_available():
                targets = targets.cuda()

            targets[torch.lt(IoU_max, 0.4), :] = 0 # lt (less then) computes IoU_max < 0.4 elementwise 

            positive_indices = torch.ge(IoU_max, 0.5) # ge (greater or equal) computes IoU_max >= 0.5 elementwise 

            num_positive_anchors = positive_indices.sum()

            assigned_annotations = bbox_annotation[IoU_argmax, :]

            targets[positive_indices, :] = 0
            targets[positive_indices, assigned_annotations[positive_indices, 4].long()] = 1

            alpha_factor = torch.ones_like(targets) * alpha
            if torch.cuda.is_available():
                alpha_factor = alpha_factor.cuda()

            alpha_factor = torch.where(torch.eq(targets, 1.), alpha_factor, 1. - alpha_factor) # torch.where(condition, x, y) 
            focal_weight = torch.where(torch.eq(targets, 1.), 1. - classification, classification) # Return a tensor of elements selected from either x or y, depending on condition.
            focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

            bce = -( targets * torch.log(classification) + (1.0 - targets) * torch.log(1.0 - classification) )

            cls_loss = focal_weight * bce # (49104, 3)

            zeros = torch.zeros_like(cls_loss)
            if torch.cuda.is_available():
                zeros = zeros.cuda()
            cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, zeros)

            classification_losses.append(cls_loss.sum() / torch.clamp(num_positive_anchors.to(dtype), min=1.0))

            if positive_indices.sum() > 0:
                assigned_annotations = assigned_annotations[positive_indices, :]

                anchor_widths_pi = anchor_widths[positive_indices]
                anchor_heights_pi = anchor_heights[positive_indices]
                anchor_ctr_x_pi = anchor_ctr_x[positive_indices]
                anchor_ctr_y_pi = anchor_ctr_y[positive_indices]

                gt_widths = assigned_annotations[:, 2] - assigned_annotations[:, 0]
                gt_heights = assigned_annotations[:, 3] - assigned_annotations[:, 1]
                gt_ctr_x = assigned_annotations[:, 0] + 0.5 * gt_widths
                gt_ctr_y = assigned_annotations[:, 1] + 0.5 * gt_heights

                # efficientdet style
                gt_widths = torch.clamp(gt_widths, min=1)
                gt_heights = torch.clamp(gt_heights, min=1)

                targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
                targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
                targets_dw = torch.log(gt_widths / anchor_widths_pi)
                targets_dh = torch.log(gt_heights / anchor_heights_pi)

                targets = torch.stack((targets_dy, targets_dx, targets_dh, targets_dw))
                targets = targets.t()

                regression_diff = torch.abs(targets - regression[positive_indices, :])

                regression_loss = torch.where(
                    torch.le(regression_diff, 1.0 / 9.0), # <= 
                    0.5 * 9.0 * torch.pow(regression_diff, 2),
                    regression_diff - 0.5 / 9.0
                )
                regression_losses.append(regression_loss.mean())
            
            else: # if there is no positive indices (overlap >= 0.5)
                if torch.cuda.is_available():
                    regression_losses.append(torch.tensor(0).to(dtype).cuda())
                else:
                    regression_losses.append(torch.tensor(0).to(dtype))

        # debug
        imgs = kwargs.get('imgs', None)


        imgs_pred = 0
        imgs_gt = 0

        if imgs is not None:
            regressBoxes = BBoxTransform()
            clipBoxes = ClipBoxes()
            obj_list = kwargs.get('obj_list', None)
            out = postprocess(imgs.detach(), # only the shape is used from imags 
                              torch.stack([anchors[0]] * imgs.shape[0], 0).detach(), regressions.detach(), classifications.detach(),
                              regressBoxes, clipBoxes,
                              threshold = 0.5, iou_threshold = 0.3)
                              
            imgs = imgs.permute(0, 2, 3, 1).cpu().numpy() # to numpy
            imgs = undo_packing(imgs)

            if self.params is None:
                mean = np.array([[0., 0., 0.]])
                std = np.array([[1., 1., 1.]])
            else:
                mean = np.array([[self.params.mean]])
                std = np.array([[self.params.std]])

                #self.mean = np.array([[mean]])
                #self.std = np.array([[std]])

            ## do not perform this since input is diff
            imgs = ((imgs * [std] + [mean]) * 255).astype(np.uint8)

            #imgs = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in imgs]

            imgs_pred = display(out, imgs.copy(), obj_list, imshow=False, imwrite=False)
            imgs_gt = display_annotations(annotations.cpu().numpy(), imgs.copy(), obj_list, imshow=False, imwrite=False)


            # anchor_boxes = torch.from_numpy(anchor_boxes.astype(dtype)).to(image.device)
            imgs_pred = torch.from_numpy(imgs_pred).to(classification.device).permute(0, 3, 1, 2)
            imgs_gt = torch.from_numpy(imgs_gt).to(classification.device).permute(0, 3, 1, 2)

        return torch.stack(classification_losses).mean(dim=0, keepdim=True), \
               torch.stack(regression_losses).mean(dim=0, keepdim=True) * 50, imgs_gt, imgs_pred  # https://github.com/google/automl/blob/6fdd1de778408625c1faf368a327fe36ecd41bf7/efficientdet/hparams_config.py#L233
