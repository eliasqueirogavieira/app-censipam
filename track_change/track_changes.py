from distutils.command.build_scripts import first_line_re

#from get_folder_changes import *
import os
import os.path
import glob
from pathlib import Path

from datetime import datetime
import hashlib

import gpt.snap_processing

# hash_file = "cur_HASHFILE.txt"
# folders = "/censipam_data/Datasets/iceye_data/dados_antigos_backup/dados_out21/iceye/"
# cur_dir = os.getcwd()

def create_dummy_file(folder, number):
	os.system(('echo File \\# %d > ' % number) + \
			"'" + folder + "'/" + ('"arq %d.txt"' % number))
 
 
def get_files_recursively(folder):
    
    root_folder = os.path.join(folder, '**/*.h5')
    files = glob.glob(root_folder, recursive=True)
    return files

def get_hash(_file):
    hasher = hashlib.md5()
    
    with open(_file, 'rb') as tmp:
        buf = tmp.read()
        hasher.update(buf)
    
    return hasher.hexdigest()


class matches(object):
    
    def __init__(self, first_image, last_image):
        self.first_image = first_image
        self.last_image = last_image
    
    def combined_names(self):
        #first = os.path.basename(self.first_image)
        #last =  os.path.basename(self.last_image)
        first = Path(self.first_image).stem
        last =  Path(self.last_image).stem
        
        #fileout = "stack_{0}_{1}".format(first, last)
        fileout = "stacked_images"
        return fileout
    
    def get_output_folder_name(self):
        first = Path(self.first_image).stem
        last =  Path(self.last_image).stem
        now = datetime.today()
        dt_string = now.strftime("%Y%m%d_%H%M%S")
        folder_out = "{0}_{1}_pred_date_{2}".format(first, last, dt_string)
        return folder_out


class image_pairs(object):

    # args.data_path, args.config, args.root_output
    def __init__(self, root_path, config, root_output):
        self.root_path = root_path
        self.config = config
        self.root_output = root_output
        self.index = 0
        self.h5files = None
        self.tif_processed = None
        
    def get_h5Files(self):
        self.h5files = get_files_recursively(self.root_path)
        self.h5files.sort()
    
    def find_pairs(self):

        self.get_h5Files()
        self.tif_processed = snap_processing.apply_snap_basic_process(self.h5files, self.config, self.root_output)
        
        
        
        #filename_stack = snap_process_image_pair(pair, args.config, outputname)
        
        # self.pairs = []
        # for i in range(0, len(all_files), 2):
        #     self.pairs.append(pair(all_files[i], all_files[i+1]))
            

        
    # def snap_process_h5totif(self):
        
        
            
    # def number_of_pairs(self):
    #     return len(self.pairs)

    # def get_next_pair(self):
        
    #     if (self.index) > len(self.pairs):
    #         return None
        
    #     tmp = self.pairs[self.index]
    #     self.index += 1

    #     return tmp
    
    # def 



# files = get_files_recursively(folders)

# get_hash(files[0])

