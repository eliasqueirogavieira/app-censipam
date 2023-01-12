import torch
import argparse
import yaml
import math
from torch import Tensor
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

from censipam import Censipam
import censipam as cspm
from skimage.io import imread
import numpy as np

from semseg.metrics import Metrics
import rasterio

from PIL import Image, ImageDraw, ImageFont

def custom_img_read(fname):

    with rasterio.open(fname) as src:
        b1 = src.read(1)
        b2 = src.read(2)

    return np.stack([b1, b2], 0)






@torch.no_grad()
def evaluate_metrics(preds, labels):

    metrics = Metrics(2, -1, 'cpu')
    metrics.update(preds, labels)
    
    ious, miou = metrics.compute_iou()
    acc, macc = metrics.compute_pixel_acc()
    f1, mf1 = metrics.compute_f1()
    
    return acc, macc, f1, mf1, ious, miou

    #return metrics.torchMetrics


class SemSeg:

    def __init__(self, cfg) -> None:
        # inference device cuda or cpu
        self.device = torch.device(cfg['DEVICE'])

        # get dataset classes' colors and labels
        self.palette = eval(cfg['DATASET']['NAME']).PALETTE
        self.palette = torch.tensor([[0], [255]])

        self.labels = eval(cfg['DATASET']['NAME']).CLASSES

        # initialize the model and load weights and send to device
        self.model = eval(cfg['MODEL']['NAME'])(cfg['MODEL']['BACKBONE'], len(self.palette)) # len(self.palette)
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
        console.print(f"Original Image Size > [red]{H}x{W}[/red]")
        # scale the short side of image to target size
        scale_factor = self.size[0] / min(H, W)
        nH, nW = round(H*scale_factor), round(W*scale_factor)
        # make it divisible by model stride
        nH, nW = int(math.ceil(nH / 32)) * 32, int(math.ceil(nW / 32)) * 32
        console.print(f"Inference Image Size > [red]{nH}x{nW}[/red]")
        # resize the image
        image = T.Resize((nH, nW))(image)
        # divide by 255, norm and add batch dim
        image = self.tf_pipeline(image).to(self.device)
        return image

    def postprocess(self, orig_img: Tensor, seg_map: Tensor, overlay: bool) -> Tensor:
        # resize to original image size
        seg_map = F.interpolate(seg_map, size=orig_img.shape[-2:], mode='bilinear', align_corners=True)
        # get segmentation map (value being 0 to num_classes)
        seg_map = seg_map.softmax(dim=1).argmax(dim=1).cpu().to(int)

        # convert segmentation map to color map
        seg_image = self.palette[seg_map].squeeze()
        if overlay: 
            seg_image = (orig_img.permute(1, 2, 0) * 0.4) + (seg_image * 0.6)

        #image = draw_text(seg_image, seg_map, self.labels)
        #return image

        seg_image = seg_image.to(torch.uint8)
        seg_image = Image.fromarray(seg_image.numpy())
        return seg_image

        

    @torch.inference_mode()
    @timer
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

    
    def predict_eval(self, img_fname: str, mask_fname: str, overlay: bool) -> Tensor:
        
        #image = torch.tensor(imread(img_fname).transpose([2, 0, 1])) 
        
        image = torch.tensor(custom_img_read(img_fname).transpose([2, 0, 1])) 

        mask = self.mask_preprocess(mask_fname) 
        
        img = self.preprocess(image)
        seg_map = self.model_forward(img)

        acc, macc, f1, mf1, ious, miou = evaluate_metrics(seg_map.softmax(1), mask)

        image = torch.concat([image, torch.zeros(1, image.shape[1], image.shape[2] )], 0)
        seg_map = self.postprocess(image, seg_map, overlay)
        return seg_map, [acc, macc, f1, mf1, ious, miou]



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/ade20k.yaml')
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    test_file = Path(cfg['TEST']['FILE'])
    mask_file = None if cfg['TEST']['LABEL'] == None else Path(cfg['TEST']['LABEL'])

    if not test_file.exists():
        raise FileNotFoundError(test_file)

    console.print(f"Model > [red]{cfg['MODEL']['NAME']} {cfg['MODEL']['BACKBONE']}[/red]")
    console.print(f"Model > [red]{cfg['DATASET']['NAME']}[/red]")

    save_dir = Path(cfg['SAVE_DIR']) / 'test_results'
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



if __name__ == '__main__':

    main()
    