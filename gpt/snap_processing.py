import os
from unittest.mock import sentinel
from config import snap_configuration
import yaml
from pathlib import Path
from os import makedirs

from shapely.geometry import Polygon
from shutil import rmtree

from glob import glob
from geopandas import GeoDataFrame, read_file

class iceye:

    @staticmethod
    def snap_basic_process(input_h5, config, output_file, output_fmt = "BEAM-DIMAP"):

        command_str = f"{config.GPT['app']} {config.GPT['xml_file_gpt_h5_preprocess']} \
                        -Pfile={input_h5} \
                        -Poutfile={output_file} -Pformat={output_fmt}"
    
        os.system(command_str)
        print("Done")

    @staticmethod
    def apply_snap_basic_process(img_path, config, output_folder, output_fmt = "BEAM-DIMAP"):
        tmp = iceye.snap_basic_process(img_path, config, output_folder, output_fmt)


    @staticmethod
    def snap_stack(img_pair, config):

        out_folder = Path(config.CMD_LINE['root_output']) / 'app_results' /\
            f'{img_pair.first_image.stem}_{img_pair.last_image.stem}'  # / 'step_2_file_'

        command_str = f"{config.GPT['app']} {config.GPT['xml_file_gpt_stack']} \
                        -Pprev={img_pair.first_image}.dim \
                        -Pafter={img_pair.last_image}.dim \
                        -Pstack={out_folder}/stacked.tif -Pformat=GeoTIFF"
        
        print("Executing ..... {0}".format(command_str) )
        os.system(command_str)
        print("Done")

        return f"{out_folder}/stacked.tif"
    

class sentinel:

    @staticmethod
    def __parse_dim_file(dim_fname):

        def get_val(coord, end):
            end = text.find(coord, end)
            start = text.find('>', end)
            end = text.find('<', start)
            val = float(text[start+1:end])
            return val, end

        with open(str(dim_fname)) as fid: text = fid.read()

        dict = [('first_near_lat','first_near_long'), ('first_far_lat', 'first_far_long'),
                ('last_near_lat', 'last_near_long'), ('last_far_lat', 'last_far_long')]
        
        pts = []
        end = 0
        for lat, long in dict:
            y, end = get_val(lat, end)
            x, end = get_val(long, end)
            pts.append([x, y])

        # fix crossing lines
        tmp = pts[2] 
        pts[2] = pts[3]
        pts[3] = tmp 
        return pts

    @staticmethod
    def __parse_dim_files(dim_fnames):
        intersecs = None
        for fi in dim_fnames:
            pts  = sentinel.__parse_dim_file(fi)
            pol = Polygon([(pts[0],pts[1]) for pts in pts])
            if intersecs == None:
                intersecs = pol
            else:
                intersecs = intersecs.union(pol)
        return intersecs

    @staticmethod
    def apply_graph_step1(in_fname, out_folder, cfg):
        
        if os.path.exists(out_folder) == False:
            makedirs(out_folder)
        
        if ( os.path.exists(out_folder / 'footprint.shp') ):

            shp = read_file(out_folder / 'footprint.shp')
            footprint = shp.geometry.values[0]
            
        else:
            
            command_str = f"{cfg.GPT['app']} {cfg.GPT['xml_step_1']} \
                    -Pin={in_fname} \
                    -Pout1={out_folder}/step_1_file_1 \
                    -Pout2={out_folder}/step_1_file_2 \
                    -Pout3={out_folder}/step_1_file_3"

            os.system(command_str)

            footprint = sentinel.__parse_dim_files([f'{out_folder}/step_1_file_1.dim', 
                                                f'{out_folder}/step_1_file_2.dim', 
                                                f'{out_folder}/step_1_file_3.dim'])

            shp = GeoDataFrame({'footprint': [str(in_fname)], 'geometry': [footprint]}, crs='EPSG:4326')
            fname = out_folder / 'footprint.shp'
            shp.to_file(str(fname))

            [rmtree(f) if os.path.isdir(f) else os.remove(f) for f in glob(str(out_folder / 'step_1_*'))]

        return footprint

    @staticmethod
    def snap_stack(img_pair, cfg):

        out_folder = Path(cfg.CMD_LINE['root_output']) / 'app_results' /\
            f'{img_pair.first_image.stem}_{img_pair.last_image.stem}'

        print(f"\t\tAppCensipam Info ~~~~ SNAP-processing pair > {out_folder.stem}")

        # STEP 1 - A
        command_str = f"{cfg.GPT['app']} {cfg.GPT['xml_step_1']} \
                    -Pin={img_pair.first_image} \
                    -Pout1={out_folder}/step_1_file_11 \
                    -Pout2={out_folder}/step_1_file_12 \
                    -Pout3={out_folder}/step_1_file_13"
        if ( not os.path.exists(f'{out_folder}/stacked.tif') ): os.system(command_str)

        # STEP 1 - B
        command_str = f"{cfg.GPT['app']} {cfg.GPT['xml_step_1']} \
                    -Pin={img_pair.last_image} \
                    -Pout1={out_folder}/step_1_file_21 \
                    -Pout2={out_folder}/step_1_file_22 \
                    -Pout3={out_folder}/step_1_file_23"
        if ( not os.path.exists(f'{out_folder}/stacked.tif') ): os.system(command_str)

        # STEP 2
        command_str = f"{cfg.GPT['app']} {cfg.GPT['xml_step_2']} \
                    -Pin1={out_folder}/step_1_file_11.dim \
                    -Pin2={out_folder}/step_1_file_21.dim \
                    -Pout={out_folder}/step_2_file_1"
        if ( not os.path.exists(f'{out_folder}/stacked.tif') ): os.system(command_str)

        [rmtree(f) if os.path.isdir(f) else os.remove(f) for f in glob(str(out_folder / 'step_1_file_11.*'))]
        [rmtree(f) if os.path.isdir(f) else os.remove(f) for f in glob(str(out_folder / 'step_1_file_21.*'))]

        command_str = f"{cfg.GPT['app']} {cfg.GPT['xml_step_2']} \
                    -Pin1={out_folder}/step_1_file_12.dim \
                    -Pin2={out_folder}/step_1_file_22.dim \
                    -Pout={out_folder}/step_2_file_2"
        if ( not os.path.exists(f'{out_folder}/stacked.tif') ): os.system(command_str)

        [rmtree(f) if os.path.isdir(f) else os.remove(f) for f in glob(str(out_folder / 'step_1_file_12.*'))]
        [rmtree(f) if os.path.isdir(f) else os.remove(f) for f in glob(str(out_folder / 'step_1_file_22.*'))]

        command_str = f"{cfg.GPT['app']} {cfg.GPT['xml_step_2']} \
                    -Pin1={out_folder}/step_1_file_13.dim \
                    -Pin2={out_folder}/step_1_file_23.dim \
                    -Pout={out_folder}/step_2_file_3"
        if ( not os.path.exists(f'{out_folder}/stacked.tif') ): os.system(command_str)

        [rmtree(f) if os.path.isdir(f) else os.remove(f) for f in glob(str(out_folder / 'step_1_file_13.*'))]
        [rmtree(f) if os.path.isdir(f) else os.remove(f) for f in glob(str(out_folder / 'step_1_file_23.*'))]
        
        # STEP 3        
        command_str = f"{cfg.GPT['app']} {cfg.GPT['xml_step_3']} \
                    -Pin1={out_folder}/step_2_file_1.dim \
                    -Pin2={out_folder}/step_2_file_2.dim \
                    -Pin3={out_folder}/step_2_file_3.dim \
                    -Pout={out_folder}/step_3_file"
        if ( not os.path.exists(f'{out_folder}/stacked.tif') ): os.system(command_str)

        [rmtree(f) if os.path.isdir(f) else os.remove(f) for f in glob(str(out_folder / 'step_2_*'))]
        # step 3A
        command_str = f"{cfg.GPT['app']} {cfg.GPT['xml_step_3_a']} \
                    -Pin={out_folder}/step_3_file.dim \
                    -Pout={out_folder}/step_3_file_a"    
        if ( not os.path.exists(f'{out_folder}/stacked.tif') ): os.system(command_str)

        # step 3B
        command_str = f"{cfg.GPT['app']} {cfg.GPT['xml_step_3_b']} \
                    -Pin={out_folder}/step_3_file.dim \
                    -Pout={out_folder}/step_3_file_b"
        if ( not os.path.exists(f'{out_folder}/stacked.tif') ): os.system(command_str)

        [rmtree(f) if os.path.isdir(f) else os.remove(f) for f in glob(str(out_folder / 'step_3_file.*'))]
        # step 4 [ BEAM-DIMAP  |  GeoTIFF-BigTIFF ]
        command_str = f"{cfg.GPT['app']} {cfg.GPT['xml_step_4']} \
                    -Pin1={out_folder}/step_3_file_a.dim \
                    -Pin2={out_folder}/step_3_file_b.dim \
                    -Pout={out_folder}/stacked.tif -Pformat=GeoTIFF-BigTIFF"

        if ( not os.path.exists(f'{out_folder}/stacked.tif') ): os.system(command_str)
        
        [rmtree(f) if os.path.isdir(f) else os.remove(f) for f in glob(str(out_folder / 'step_3_file*'))]

        print(f"\t\tAppCensipam Info ~~~~ SNAP-processing pair > {out_folder.stem} has ended")
        
        return f"{out_folder}/stacked.tif"

    


def snap_process(pair, config):

    gpt_class = eval(config.IN_DATA['format'])
    stacked_fname = gpt_class.snap_stack(pair, config)
    assert os.path.exists(stacked_fname), f'Stacked file processing ended up unsuccessful'
    with open( Path(config.CMD_LINE.root_output) / 'exec_report.txt', 'a') as report:
        report.write(f'Stacked match: {Path(stacked_fname).parts[-2]}  > Ok \n ')

    return stacked_fname
