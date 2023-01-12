from email import utils
from matplotlib import patches
import torch
import argparse
import yaml
import math
from torch import Tensor, tensor
from torch.nn import functional as F
from pathlib import Path
from torchvision import io
from torchvision import transforms as T
from semseg.models import *
from semseg.datasets import *
from semseg.utils.utils import timer
from semseg.utils.visualize import draw_text

from rich.console import Console
console = Console()

from skimage.io import imread
import numpy as np

from semseg.metrics import Metrics
import rasterio

from utils import Metrics as MetricsNp

from PIL import Image, ImageDraw, ImageFont
from torchmetrics.functional import f1_score, dice, jaccard_index, confusion_matrix, stat_scores
import utils
from censipam import Censipam
import censipam as cspm

def custom_img_read(fname):

    with rasterio.open(fname) as src:
        b1 = src.read(1)
        b2 = src.read(2)

    return np.stack([b1, b2], 0)

@torch.no_grad()
def evaluate_metrics(preds, labels):

    #metrics = Metrics(2, -1, 'cpu')
    metrics = MetricsNp(1, -1, 'cuda')
    metrics.update(preds, labels)
    
    #ious, miou = metrics.compute_iou()
    
    miou = metrics.compute_iou_defor()
    #acc, macc = metrics.compute_pixel_acc()
    #f1, mf1 = metrics.compute_f1()

    prec = metrics.precision()
    rec = metrics.recall()

    mf1 = (2*prec*rec) / (prec + rec)
    mf1 = mf1.round(2) 
    
    #return acc, macc, f1, mf1, ious, miou
    return prec, rec, mf1, miou, metrics.nb_val_samples

    #return metrics.torchMetrics


class SemSeg:

    def __init__(self, cfg) -> None:
        # inference device cuda or cpu
        self.device = torch.device(cfg['DEVICE'])

        # get dataset classes' colors and labels
        self.palette = eval(cfg['DATASET']['NAME']).PALETTE
        self.palette = torch.tensor([[128], [255]])

        self.labels = eval(cfg['DATASET']['NAME']).CLASSES

        # initialize the model and load weights and send to device
        self.model = eval(cfg['MODEL']['NAME'])() # len(self.palette)
        self.model.load_state_dict(torch.load(cfg['TEST']['MODEL_PATH'], map_location='cpu'))
        self.model = self.model.to(self.device)
        self.model.eval()

        # preprocess parameters and transformation pipeline
        self.size = cfg['TEST']['IMAGE_SIZE']
        
        self.tf_pipeline = T.Compose([
            T.Lambda(lambda x: x / 255),
            T.Lambda(lambda x: x.unsqueeze(0)),
        ])

    
    def mask_preprocess(self, mask):
        #mask = torch.tensor(np.expand_dims(imread(mask_fname), 0))
        label = np.expand_dims( imread(mask), 0 )
        label[label == 0] = 1
        label[label == 255] = 2
        label = torch.tensor(label).type(torch.int16)
        return label - 1

    def preprocess(self, image: Tensor) -> Tensor:
        H, W = image.shape[1:]
        #console.print(f"Original Image Size > [red]{H}x{W}[/red]")
        # scale the short side of image to target size
        #scale_factor = self.size[0] / min(H, W)
        scale_factor = 1.0

        nH, nW = round(H*scale_factor), round(W*scale_factor)
        # make it divisible by model stride
        nH, nW = int(math.ceil(nH / 32)) * 32, int(math.ceil(nW / 32)) * 32
        #console.print(f"Inference Image Size > [red]{nH}x{nW}[/red]")
        # resize the image
        image = T.Resize((nH, nW))(image)
        # divide by 255, norm and add batch dim
        image = self.tf_pipeline(image).to(self.device)
        return image

    def postprocess(self, orig_img: Tensor, seg_map: Tensor, overlay: bool) -> Tensor:
        # resize to original image size
        seg_map = F.interpolate(seg_map, size=orig_img.shape[-2:], mode='bilinear', align_corners=True)
        seg_map = torch.sigmoid(seg_map)
        
        seg_map = (seg_map>0.5).cpu().to(int)
        seg_image = self.palette[seg_map].squeeze()
        if overlay: 
            seg_image = (orig_img.permute(1, 2, 0) * 0.4) + (seg_image * 0.6)

        seg_image = seg_image.to(torch.uint8)
        seg_image = Image.fromarray(seg_image.numpy())
        return seg_image

        

    @torch.inference_mode()
    #@timer
    def model_forward(self, img: Tensor) -> Tensor:
        return self.model(img)
        
    def predict(self, img_fname: str, overlay: bool) -> Tensor:
        #image = torch.tensor(imread(img_fname).transpose([2, 0, 1])) # image = io.read_image(img_fname)
        image = torch.tensor(custom_img_read(img_fname)) # image = io.read_image(img_fname)

        img = self.preprocess(image)
        seg_map = self.model_forward(img)
        image = torch.concat([image, torch.zeros(1, image.shape[1], image.shape[2] )], 0)
        seg_map = self.postprocess(image, seg_map, overlay)
        return seg_map

    
    def predict_eval(self, in_img: str, in_label: str, overlay: bool) -> Tensor:
        
        #image = torch.tensor(imread(img_fname).transpose([2, 0, 1])) 
        
        image = torch.tensor(custom_img_read(in_img).transpose([2, 0, 1])) 
        label_mask = self.mask_preprocess(in_label) 
        
        img = self.preprocess(image)
        seg_map = self.model_forward(img)

        precision, recall, mf1, miou = evaluate_metrics(seg_map.softmax(1), label_mask)

        #return prec, rec, mf1, miou

        image = torch.concat([image, torch.zeros(1, image.shape[1], image.shape[2] )], 0)
        seg_map = self.postprocess(image, seg_map, overlay)
        return seg_map, [precision, recall, mf1, miou]

    
    def predict_eval_img(self, in_img, label_mask)-> Tensor:
                
        label_mask[label_mask == 128] = 1
        label_mask[label_mask == 255] = 2
        label_mask = torch.tensor(label_mask).type(torch.int16)
        label_mask = label_mask - 1
        label_mask = label_mask.to(self.device)

        image = torch.tensor(in_img) 
        img = self.preprocess(image)
        seg_map = self.model_forward(img)

        probs = seg_map.softmax(1)

        prec, rec, mf1, miou, valid_samples = evaluate_metrics(probs, label_mask)

        probs = probs[0, 1, ...].cpu().numpy()

        image = torch.concat([image, torch.zeros(1, image.shape[1], image.shape[2] )], 0)
        seg_map = self.postprocess(image, seg_map, False)

        seg_map = np.array(seg_map)

        return seg_map, probs, [prec, rec, mf1, miou, valid_samples]
    

    def compute_metric(self, label, pred_mask):

        metrics = MetricsNp(1, -1, self.device)
        
        label = label.astype(np.int16)
        label[label == 128] = 1
        label[label == 255] = 2
        label = label - 1

        pred_mask = pred_mask.astype(np.int16)
        pred_mask[pred_mask == 128] = 0 # 1
        pred_mask[pred_mask == 255] = 1 # 2
        #pred_mask = pred_mask - 1

        metrics.update(pred_mask.flatten(), label.flatten())
        miou = metrics.compute_iou_defor()
        #acc, macc = metrics.compute_pixel_acc()
        #f1, mf1 = metrics.compute_f1()

        prec = metrics.precision()
        rec = metrics.recall()

        mf1 = 2*prec*rec / (prec + rec)
        mf1 = mf1.round(2)


        print(f"Global -- prec: {prec} \t rec: {rec} \t f1: {mf1} \t iou: {miou}")


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-model',  type=str)
    parser.add_argument('-output',  type=str)
    parser.add_argument('--cfg', type=str, default='configs/ade20k.yaml')
    
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    cfg['TEST']['MODEL_PATH'] = args.model
    cfg['SAVE_DIR'] = args.output

    test_file = Path(cfg['TEST']['FILE'])
    mask_file = None if cfg['TEST']['LABEL'] == None else Path(cfg['TEST']['LABEL'])

    if not test_file.exists():
        raise FileNotFoundError(test_file)

    console.print(f"Model > [red]{cfg['MODEL']['NAME']} {cfg['MODEL']['BACKBONE']}[/red]")
    console.print(f"Model > [red]{cfg['DATASET']['NAME']}[/red]")

    #save_dir = Path(cfg['SAVE_DIR']) / 'test_results'
    save_dir = Path(cfg['SAVE_DIR'])
    save_dir.mkdir(exist_ok=True)
    
    semseg = SemSeg(cfg)

    with console.status("[bright_green]Processing..."):

        if test_file.is_file():
        
            console.rule(f'[green]{test_file}')
            segmap = semseg.predict(str(test_file), cfg['TEST']['OVERLAY'])
            segmap.save(save_dir / f"{str(test_file.stem)}.png")
        
        elif mask_file:

            files = list(test_file.glob('*.*')) # files = list(test_file.glob('*.*'))
            files_mask = list(mask_file.glob("*.png"))

            files.sort()
            files_mask.sort()

            for img_fname, mask_fname in zip(files, files_mask):
                console.rule(f'[green]{img_fname}')
                #segmap = semseg.predict(str(img_fname), cfg['TEST']['OVERLAY'])
                segmap, metrics = semseg.predict_eval(str(img_fname), str(mask_fname), cfg['TEST']['OVERLAY'])
                segmap.save(save_dir / f"{str(img_fname.stem)}.png")

                # acc, macc, f1, mf1, ious, miou
                print("{} \t metrics - macc = {} | mf1 = {} | miou = {} ".format(img_fname.stem, metrics[1], metrics[3], metrics[5]))

        else:
            
            files = list(test_file.glob('*.tif'))
            files.sort()

            for img_fname in files:
                console.rule(f'[green]{img_fname}')
                segmap = semseg.predict(str(img_fname), cfg['TEST']['OVERLAY'])
                segmap.save(save_dir / f"{str(img_fname.stem)}.png")

                # acc, macc, f1, mf1, ious, miou
                #print("{} \t metrics - macc = {} | mf1 = {} | miou = {} ".format(img_fname.stem, metrics[1], metrics[3], metrics[5]))


    console.rule(f"[cyan]Segmentation results are saved in `{save_dir}`")


def pre_process(test_file):

    import utils
    patches = utils.decompose_image(test_file, patch_size = 512, offset = (256, 256))

    with rasterio.open(test_file) as src:
        meta = src.meta
        vv = src.read(1)
        label_mask = src.read(3)
    
    return [vv, label_mask, meta], patches


def process_whole():

    parser = argparse.ArgumentParser()
    parser.add_argument('-model',  type=str, default='/censipam_data/eliasqueiroga/es_best_model_sd.pth')
    parser.add_argument('-output',  type=str, default='/censipam_data/eliasqueiroga/app_novo/infer')
    parser.add_argument('--cfg', type=str, default='/censipam_data/eliasqueiroga/app_novo/configs/censipam_whole.yaml')
    
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    test_file = Path(cfg['TEST']['FILE'])

    if not test_file.exists():
        raise FileNotFoundError(test_file)
    

    with rasterio.open(test_file) as src:
        meta = src.meta
        vv = src.read(1)
        label_mask = src.read(3)
    
    patches_0_0 = utils.decompose_image(test_file, patch_size = 512, offset = (0, 0))
    patches_0_256 = utils.decompose_image(test_file, patch_size = 512, offset = (0, 256))
    patches_256_0 = utils.decompose_image(test_file, patch_size = 512, offset = (256, 0))
    patches_256_256 = utils.decompose_image(test_file, patch_size = 512, offset = (256, 256))
    
    patches_1024 = utils.decompose_image(test_file, patch_size = 1024, offset = (0, 0))
    patches_1024_512 = utils.decompose_image(test_file, patch_size = 1024, offset = (512, 512))

    grids = [patches_0_0, patches_0_256, patches_256_0, patches_256_256] # patches_1024, patches_1024_512
    
    preds  = []
    weights  = []

    for grid in grids:

        p, w = run_inf_whole(cfg, grid, label_mask)
        preds.append(p)
        weights.append(w)

    # meta['count'] = 4
    # meta['dtype'] = rasterio.float32
    # with rasterio.open('/censipam_data/renam/cpred6.tif', 'w', **meta) as dst:
    #     dst.write_band(1, preds[0])   
    #     dst.write_band(2, preds[1])   
    #     dst.write_band(3, preds[2])   
    #     dst.write_band(4, preds[3])   

    # print("\n\n Combined")

    # #merge_preds_cnn(cfg, label_mask, preds, weights)
    
    merge_preds(cfg, label_mask, preds, weights, meta) ## weighted avg
    # merge_preds_avg(cfg, label_mask, preds, weights) # simple avg
    # merge_preds_sum(cfg, label_mask, preds, weights) # sum 

    # merge_preds_v2(cfg, label_mask, preds, weights) # windowed - weighted avg
    
    #merge_preds(cfg, label_mask, [preds[0], preds[-1]], [weights[0], weights[-1]])
    #merge_preds(cfg, label_mask, [preds[1], preds[-1]], [weights[1], weights[-1]])
    #merge_preds(cfg, label_mask, [preds[2], preds[-1]], [weights[2], weights[-1]])
    
    #merge_preds(cfg, label_mask, [preds[0], preds[1]], [weights[1], weights[1]])
    #all = 1

    [pred_mask, label_mask, meta], patches = pre_process(test_file)

    
        
    meta['count'] = 1
    with rasterio.open("full_pred.tif", 'w', **meta) as dst:
        dst.write_band(1, pred_mask)

    
    # meta['count'] = 1
    # meta['dtype'] = rasterio.float32
    # meta['nodata'] = -1.0
    # with rasterio.open("score.tif", 'w', **meta) as dst:
    #     dst.write_band(1, confidence_mask)
    
    #semseg.compute_metric(label_mask, pred_mask)

    # prec_blocks = np.asarray(prec_blocks)
    # rec_blocks = np.asarray(rec_blocks)
    # f1_blocks = np.asarray(f1_blocks)
    # iou_blocks = np.asarray(iou_blocks)

    # prec_blocks = prec_blocks[~np.isnan(prec_blocks)].mean().round(2)
    # rec_blocks = rec_blocks[~np.isnan(rec_blocks)].mean().round(2)
    # f1_blocks = f1_blocks[~np.isnan(f1_blocks)].mean().round(2)

    # iou_blocks = iou_blocks[~np.isnan(iou_blocks)].mean().round(2)

    # print(f"Blocks -- prec: {prec_blocks} \t rec: {rec_blocks} \t f1: {f1_blocks} \t iou: {iou_blocks}")
    # #print("Blocks -- {:.2} \t {:.2} \t {:.2} \t {:.2}".format(prec_blocks, rec_blocks, f1_blocks, iou_blocks))

    # console.rule(f"[cyan]Segmentation results are saved in `{save_dir}`")


def merge_preds(cfg, label_mask, preds, weights, meta):

    preds = np.stack(preds)
    weights = np.stack(weights)

    #mask = (preds == -1).sum(axis=0) >= 1
    mask = (preds == -1).sum(axis=0) == 4
    p = (preds*weights).sum(axis=0) / weights.sum(axis=0)

    p[ p > 0.4] = 255
    p[ p <= 0.4] = 128
    p[mask] = 0
    
    p = p.astype(np.uint8)
    p_m = morph(p)

    meta['count'] = 1
    with rasterio.open("full_pred.tif", 'w', **meta) as dst:
        dst.write_band(1, p)


    semseg = SemSeg(cfg)
    semseg.compute_metric(label_mask, p)
    semseg.compute_metric(label_mask, p_m)
    print('\n')


def merge_preds_cnn(cfg, label_mask, preds, weights):

    #torch.Size([1, 4, 21668, 20565])

    in_preds = np.stack(preds) 
    
    in_preds = np.expand_dims(in_preds, 0) # [Batch, Preds, H, W]
    in_preds = torch.tensor(in_preds)


    from merge_preds import mergeNet
    model_fname = '/censipam_data/eliasqueiroga/es_bestmodel_sd.pth'

    model = mergeNet()
    model.load_state_dict(torch.load(model_fname, map_location='cpu'))
    model.eval()

    logits = model(in_preds)

    print('Test')






def morph(pred):
    
    from skimage.morphology import opening, square, closing, erosion, dilation

    mask = pred > 0
    pred [ pred < 255] = 0
    pred [ pred == 255] = 1

    pred = erosion(pred, square(7))
    pred = dilation(pred, square(11))

    pred[ pred == 0 ] = 128
    pred[ pred == 1 ] = 255
    pred[ ~ mask ] = 0

    return pred

def merge_preds_avg(cfg, label_mask, preds, weights):

    preds = np.stack(preds)
    weights = np.stack(weights)

    #mask = (preds == -1).sum(axis=0) >= 1
    mask = (preds == -1).sum(axis=0) == 4
    p = preds.mean(axis=0)
    
    p[ p > 0.45] = 255
    p[ p <= 0.45] = 128
    p[mask] = 0
    p = p.astype(np.uint8)

    p_m = morph(p)

    semseg = SemSeg(cfg)
    semseg.compute_metric(label_mask, p)
    semseg.compute_metric(label_mask, p_m)
    print('\n')

def merge_preds_sum(cfg, label_mask, preds, weights):

    preds = np.stack(preds)
    weights = np.stack(weights)

    #mask = (preds == -1).sum(axis=0) >= 1
    mask = (preds == -1).sum(axis=0) == 4

    preds[preds == -1] = 0
    p = preds.sum(axis=0)
    
    p[ p > (3*0.5)] = 255
    p[ p <= (3*0.5)] = 128
    p[mask] = 0
    p = p.astype(np.uint8)

    p_m = morph(p)

    semseg = SemSeg(cfg)
    semseg.compute_metric(label_mask, p)
    semseg.compute_metric(label_mask, p_m)
    print('\n')

def merge_preds_v2(cfg, label_mask, preds, weights):

    preds = np.stack(preds)
    weights = np.stack(weights)

    mask = (preds == -1).sum(axis=0) == 4

    # a = np.pad(preds, pad_width=((0,0),(2,2), (2,2)), constant_values=128)
    # w = np.lib.stride_tricks.sliding_window_view(a, [4,3,3]).squeeze()
    # k = w[:, :, :, 1, 1].transpose(2, 0, 1)
    # k = k[:, 1:-1, 1:-1]
    # e = preds - k

    w = np.pad(weights, pad_width=((0,0),(1,2), (1,2)), constant_values=128)
    p = np.pad(preds, pad_width=((0,0),(1,2), (1,2)), constant_values=128)
    
    w = np.lib.stride_tricks.sliding_window_view(w, [4,3,3]).squeeze()
    p = np.lib.stride_tricks.sliding_window_view(p, [4,3,3]).squeeze()

    p = np.sum(p*w, axis=(2,3,4)) / np.sum(w, axis=(2,3,4))
    #p = p[1:-1, 1:-1]
    p = p[0:-1, 0:-1]

    p[ p > 0.4] = 255
    p[ p <= 0.4] = 128
    p[mask] = 0
    p = p.astype(np.uint8)

    p_m = morph(p)

    semseg = SemSeg(cfg)
    semseg.compute_metric(label_mask, p)
    semseg.compute_metric(label_mask, p_m)
    print('\n')
    

def run_inf_whole(cfg, patches, label_mask):

    pred_mask = label_mask * 0
    confidence_mask = pred_mask.astype(np.float32) - 1

    g_weight = confidence_mask * 0 + 1e-12
    
    console.print(f"Model > [red]{cfg['MODEL']['NAME']} {cfg['MODEL']['BACKBONE']}[/red]")
    console.print(f"Model > [red]{cfg['DATASET']['NAME']}[/red]")

    save_dir = Path(cfg['SAVE_DIR'])
    save_dir.mkdir(exist_ok=True)
    
    semseg = SemSeg(cfg)

    # prec_blocks = []
    # rec_blocks = []
    # f1_blocks = []
    # iou_blocks = []
    # valid_samples = []



    with console.status("[bright_green]Processing..."):

        for patch in zip(patches):
            
            img = patch[0][0]
            label = patch[0][1]
            w_tmp = patch[0][-1]

            segmap, probs, metrics = semseg.predict_eval_img(img, label)
            assert not (0 in np.unique(segmap)), "Should not happen"

            pred_mask[w_tmp.row_off:(w_tmp.row_off + w_tmp.height), w_tmp.col_off :(w_tmp.col_off + w_tmp.width)] = segmap
            confidence_mask[w_tmp.row_off:(w_tmp.row_off + w_tmp.height), w_tmp.col_off :(w_tmp.col_off + w_tmp.width)] = probs

            w = get_weigth(w_tmp.height)
            g_weight[w_tmp.row_off:(w_tmp.row_off + w_tmp.height), w_tmp.col_off :(w_tmp.col_off + w_tmp.width)] = w

            # prec_blocks.append(metrics[0])
            # rec_blocks.append(metrics[1])
            # f1_blocks.append(metrics[2])
            # iou_blocks.append(metrics[3])
            # valid_samples.append(metrics[4])
    
    semseg.compute_metric(label_mask, pred_mask)

    return confidence_mask, g_weight



def get_weigth(block_size):
    
    from scipy.stats import multivariate_normal
    x,y = np.mgrid[-1:1:2/block_size, -1:1:2/block_size]
    pos = np.dstack((x,y))

    fx = multivariate_normal(mean=[0, 0], cov = [[1, 0], [0, 1]])
    vals = fx.pdf(pos)
    return vals


if __name__ == '__main__':

    process_whole()
    #main()
    