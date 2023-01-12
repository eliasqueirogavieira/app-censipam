
### This software was developed by a team of student, researchers and professors 
# from Universidade de Brasilia in partnership with CENSIPAM

import re
import argparse
import os.path

from matplotlib.path import Path

from models.model_hub import load_model

from folder_tracker import create_folder_analysis
from gpt.snap_processing import snap_process
from data import pre_process

from omegaconf import DictConfig, OmegaConf
from geopandas.tools import overlay
from pathlib import Path



def main(config):

    ################## find images pair to run the inference model    
    folder_tracker = create_folder_analysis(config)
    matches = folder_tracker.get_matches()

    ## load model
    model = load_model(config) 
    orig_folder = args.CMD_LINE.data_path
    ############################
    #### go over each image pair
    print(f"AppCensipam Info ~~~~ ({matches.size()}) zip pairs have been found and will be processed by the model")
    for i in range(matches.size()):

        pair = matches.item(i)
        pair.analysed = False
        if (pair.analysed): continue

        print(f'\tProcessing pair: {pair.first_image.stem} - {pair.last_image.stem}') 

        stacked_fname = snap_process(pair, config)
        out_folder = Path(stacked_fname).parent

        geo_patches_1 = pre_process(stacked_fname, config, patch_size=config.MODEL.patch_size)
        geo_patches_2 = pre_process(stacked_fname, config, patch_size=config.MODEL.patch_size, offset = (0, 256))
        geo_patches_3 = pre_process(stacked_fname, config, patch_size=config.MODEL.patch_size, offset = (256, 0))
        geo_patches_4 = pre_process(stacked_fname, config, patch_size=config.MODEL.patch_size, offset = (256, 256))

        ##############################  Run inference     
        model.predict([geo_patches_1, geo_patches_2, geo_patches_3, geo_patches_4], stacked_fname, out_folder, orig_folder)
        pair.analysed = True

        folder_tracker.serialize()

    ######### save object state to file
    


def load_cfg(config):

    cfg = OmegaConf.load(config)
    assert os.path.exists(cfg.MODEL['config']), f'Model config does not exist'
    cfg.MODEL = OmegaConf.load(cfg.MODEL['config'])
    return cfg


def parse_params():

    parser = argparse.ArgumentParser('Inference')
    parser.add_argument('--data_path', type=str, default='/censipam_data/Datasets/sentinel_data/Sentinel_1A_10_615_21Mai2022_26Jun2022', help='the root folder of dataset')
    parser.add_argument('--model', type=str, default='/censipam_data/eliasqueiroga/es_best_model_sd.pth', help='configuration file')
    parser.add_argument('--config', type=str, default='/censipam_data/eliasqueiroga/app_novo/configs/cfg_sentinel.yml', help='configuration file')
    parser.add_argument('--root_output', type=str, default='/censipam_data/eliasqueiroga/app_novo', help='Root folder for output')
    args = parser.parse_args()

    load_cfg(args.config)
    
    config_obj = load_cfg(args.config)
    config_obj['CMD_LINE'] = args.__dict__    
    
    return config_obj


if __name__ == '__main__':

    args = parse_params()
    main(args)