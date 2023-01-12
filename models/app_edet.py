import torch
import numpy as np
import geopandas
import cv2

from os.path import exists

from skimage.io import imsave
from torchvision.ops.boxes import batched_nms
from shapely.geometry import Polygon


from .edet.backbone import EfficientDetBackbone 
from .edet.utils import BB_STANDARD_COLORS, standard_to_bgr, get_index_label
from .imodel import NNModel


color_list = standard_to_bgr(BB_STANDARD_COLORS, excluded=0)
obj_list = ['deforestation']
compound_coef = 0



class BBoxTransform(torch.nn.Module):

    def forward(self, anchors, regression):
        """
        decode_box_outputs adapted from https://github.com/google/automl/blob/master/efficientdet/anchors.py

        Args:
            anchors: [batchsize, boxes, (y1, x1, y2, x2)]
            regression: [batchsize, boxes, (dy, dx, dh, dw)]

        Returns:

        """
        y_centers_a = (anchors[..., 0] + anchors[..., 2]) / 2
        x_centers_a = (anchors[..., 1] + anchors[..., 3]) / 2
        ha = anchors[..., 2] - anchors[..., 0]
        wa = anchors[..., 3] - anchors[..., 1]

        w = regression[..., 3].exp() * wa # exp(reg) * wa
        h = regression[..., 2].exp() * ha # exp(reg) * ha

        y_centers = regression[..., 0] * ha + y_centers_a
        x_centers = regression[..., 1] * wa + x_centers_a

        ymin = y_centers - h / 2.
        xmin = x_centers - w / 2.
        ymax = y_centers + h / 2.
        xmax = x_centers + w / 2.

        return torch.stack([xmin, ymin, xmax, ymax], dim=2)
    
class ClipBoxes(torch.nn.Module):

    def __init__(self):
        super(ClipBoxes, self).__init__()

    def forward(self, boxes, img):
        batch_size, num_channels, height, width = img.shape

        boxes[:, :, 0] = torch.clamp(boxes[:, :, 0], min=0)
        boxes[:, :, 1] = torch.clamp(boxes[:, :, 1], min=0)

        boxes[:, :, 2] = torch.clamp(boxes[:, :, 2], max=width - 1)
        boxes[:, :, 3] = torch.clamp(boxes[:, :, 3], max=height - 1)

        return boxes
    


def plot_one_box(img, coord, label=None, score=None, color=None, l_thickness=None):

    l_thickness = l_thickness or int(round(0.001 * max(img.shape[0:2])))  # line thickness
    l_thickness = max(l_thickness, 2)
    #color = (255, 255, 255) # white
    c1, c2 = (int(coord[0]), int(coord[1])), (int(coord[2]), int(coord[3]))
    cv2.rectangle(img, c1, c2, color, thickness=l_thickness)
    if label:
        f_thick = max(l_thickness - 2, 1)  # font thickness
        s_size = cv2.getTextSize(str('{:.0%}'.format(score)), 1, fontScale=float(l_thickness) / 3, thickness=f_thick)[0]
        t_size = cv2.getTextSize(label, 1, fontScale=float(1), thickness=f_thick)[0] # (l_thickness) / 3
        c2 = c1[0] + t_size[0] + s_size[0] , c1[1] - t_size[1] - 3

        cv2.putText(img, '{:.0%}'.format(score), (c1[0], c1[1] - 2), 1, 1, color,
                    thickness=1, lineType=cv2.FONT_HERSHEY_SIMPLEX)
                    # thickness=f_thick

def flatten_preds(list_of_list):

    all = []
    for i, list in enumerate(list_of_list):
        for k, item in enumerate(list):
            all.append(item)
    return all


def imcoor2geocoor(preds_per_img, transforms):

    polygons = []
    scores = []

    for i, img_pred in enumerate(preds_per_img):

        bbox = img_pred['rois']
        if len(bbox) < 1:
            continue

        for k, pred in enumerate(bbox):
            t_affine = transforms[i]

            p1 = (t_affine*(bbox[k][0], bbox[k][1]))
            p2 = (t_affine*(bbox[k][0], bbox[k][1] + bbox[k][3]))
            p3 = (t_affine*(bbox[k][0] + bbox[k][2], bbox[k][1] + bbox[k][3]))
            p4 = (t_affine*(bbox[k][0] + bbox[k][2], bbox[k][1]))

            polygons.append(Polygon([p1, p2, p3, p4]))
            scores.append(img_pred['scores'][k])

    return polygons, scores

def to_tensor(patches, config):

    patches = np.stack(patches, axis = 0)
    patches = torch.from_numpy(patches)

    if config.MODEL['use_cuda']:
        patches = patches.cuda(0)

    if config.MODEL['use_float16']:
        patches = patches.half()
    else:
        patches = patches.float()

    return patches


def invert_affine(preds, metas=(512, 512, 512, 512, 0 , 0)):
    for i in range(len(preds)):
        if len(preds[i]['rois']) == 0:
            continue
        else:
            if metas is float:
                preds[i]['rois'][:, [0, 2]] = preds[i]['rois'][:, [0, 2]] / metas
                preds[i]['rois'][:, [1, 3]] = preds[i]['rois'][:, [1, 3]] / metas
            else:
                new_w, new_h, old_w, old_h, padding_w, padding_h = metas
                preds[i]['rois'][:, [0, 2]] = preds[i]['rois'][:, [0, 2]] / (new_w / old_w)
                preds[i]['rois'][:, [1, 3]] = preds[i]['rois'][:, [1, 3]] / (new_h / old_h)
    return preds

def display(preds, imgs, batch_idx, imwrite=False, output_folder='tmp'):

    for i in range(len(imgs)):
        if len(preds[i]['rois']) == 0:
            continue

        idx = batch_idx * len(imgs) + i
        filename_idx = f"{idx}"
        img_tmp = imgs[i].copy()

        for j in range(len(preds[i]['rois'])):
            x1, y1, x2, y2 = preds[i]['rois'][j].astype(int)

            if x2 <= (x1 + 2):
                continue
            if y2 <= (y1 + 2):
                continue

            obj = obj_list[preds[i]['class_ids'][j]]
            score = float(preds[i]['scores'][j])
            
            if (imwrite):
                plot_one_box(img_tmp, [x1, y1, x2, y2], label=obj,score=score,color=color_list[get_index_label(obj, obj_list)])
                output_name = f'{output_folder}/img{compound_coef}_{filename_idx}.png'
                imsave(output_name, img_tmp)


def postprocess(x, anchors, regression, classification, regressBoxes, clipBoxes, threshold, iou_threshold):

    transformed_anchors = regressBoxes(anchors, regression)
    transformed_anchors = clipBoxes(transformed_anchors, x)
    scores = torch.max(classification, dim=2, keepdim=True)[0] # class with max prob
    scores_over_thresh = (scores > threshold)[:, :, 0]
    out = []

    # go over each image
    for i in range(x.shape[0]):
        if scores_over_thresh[i].sum() == 0: # in case encounter nothing
            out.append({
                'rois': np.array(()),
                'class_ids': np.array(()),
                'scores': np.array(()),
            })
            continue

        classification_per = classification[i, scores_over_thresh[i, :], ...].permute(1, 0)
        transformed_anchors_per = transformed_anchors[i, scores_over_thresh[i, :], ...]
        scores_per = scores[i, scores_over_thresh[i, :], ...]
        scores_, classes_ = classification_per.max(dim=0)
        anchors_nms_idx = batched_nms(transformed_anchors_per, scores_per[:, 0], classes_, iou_threshold=iou_threshold)

        if anchors_nms_idx.shape[0] != 0:
            classes_ = classes_[anchors_nms_idx]
            scores_ = scores_[anchors_nms_idx]
            boxes_ = transformed_anchors_per[anchors_nms_idx, :]

            out.append({
                'rois': boxes_.cpu().numpy(),
                'class_ids': classes_.cpu().numpy(),
                'scores': scores_.cpu().numpy(),})
        else:
            out.append({
                'rois': np.array(()),
                'class_ids': np.array(()),
                'scores': np.array(()),})

    return out




class ModelEdet(NNModel):


    def __init__(self, config) -> None:
        super().__init__()

        self.config = config
        
        self.compound_coef = config.MODEL['compound_coef'] # compound_coef

        self.obj_list = config.MODEL['obj_list']
        self.num_classes = len(self.obj_list) # len(obj_list)

        self.in_channels = config.MODEL['nb_in_channels']
        self.anchors_ratios =  eval(config.MODEL['anchors_ratios']) 
        self.anchors_scales = eval(config.MODEL['anchors_scales'])

        self.use_cuda = config.MODEL['use_cuda']
        self.use_float16 = config.MODEL['use_float16']

        self.threshold_score =  config.MODEL['threshold_score']
        self.nms_threshold =  config.MODEL['nms_threshold']  # non-maximal supression

        self.batch_size = config.MODEL['batch_size']

        self.checkpoint = config.CMD_LINE['model'] if exists(config.CMD_LINE['model']) else config.MODEL['checkpoint']


    def load(self):
        self.__load_model_dict()
	
    def predict_single(self, patches):

        self.__run_model(patches)
        


    def load_model(path_saved_model):

        model = torch.load(path_saved_model)
        model.requires_grad_(False)
        model.eval()

        use_cuda = True
        use_float16 = False
        if use_cuda:
            model.cuda(0)

        if use_float16:
            model.half()

        return model


    def __load_model_dict(self):

        self.model = EfficientDetBackbone(compound_coef=self.compound_coef, 
                                     num_classes=self.num_classes, 
                                     in_channels=self.in_channels,
                                     ratios=self.anchors_ratios, 
                                     scales=self.anchors_scales)
        
        assert exists(self.checkpoint), f'App efficient det: checkpoint does not exist'

        self.model.load_state_dict(torch.load(self.checkpoint, map_location=torch.device('cpu')))

        self.model.requires_grad_(False)
        self.model.eval()

        if self.use_cuda:
            self.model.cuda(0)

        if self.use_float16:
            self.model.half()

        #return model

    def __run_model(self, patches):

        batch_size = self.batch_size
        nb_batch = np.ceil(len(patches.patches) / batch_size).astype(int)
        offset_batch = 0

        filtered_scores = []
        filtered_bboxs = []
        
        for batch_idx in range(nb_batch):

            offset_batch = batch_idx * self.batch_size

            if (len(patches.patches) < self.batch_size):
                block_width = len(patches.patches)
            else:
                block_width = min(self.batch_size, len(patches.patches) - self.batch_size)

            batch = patches.patches[offset_batch: offset_batch + block_width]
            batch = to_tensor(batch, self.config)

            transforms = patches.transforms[offset_batch: offset_batch + block_width]
            _, regression, classification, anchors = self.model(batch)

            regressBoxes = BBoxTransform()
            clipBoxes = ClipBoxes()

            preds = postprocess(batch,      # only shape is used
                                anchors, regression, classification,
                                regressBoxes, clipBoxes,
                                self.threshold_score, self.nms_threshold)

            tmp_bbox, tmp_score = imcoor2geocoor(preds, transforms)
            if len(tmp_score) > 0:
                filtered_bboxs.append(tmp_bbox)
                filtered_scores.append(tmp_score)

            batch = batch.detach().cpu().numpy()
            batch = batch.transpose([0, 2, 3, 1])
            batch_uint8 = np.round(batch * (2**8-1)).astype(np.uint8)

            display(preds, batch_uint8, batch_idx, imwrite=True, output_folder= output_folder )

        filtered_scores = flatten_preds(filtered_scores)
        filtered_bboxs = flatten_preds(filtered_bboxs)

        all = {'scores': filtered_scores, 'geometry': filtered_bboxs}
        gdf = geopandas.GeoDataFrame(all, crs="EPSG:4326")

        file_preds = f'{output_folder}/preds.shp'
        gdf.to_file(file_preds)