from copyreg import pickle
import os
from select import select
import pandas as pd
from data import iceye
from footprint import *

from pathlib import Path
from datetime import datetime
import pickle
import itertools
from abc import ABC, abstractmethod

from track_change.track_changes import matches
from get_folder_changes import *
from gpt import snap_processing


class matches(object):

    def __init__(self, first_image, last_image):
        self.images = [first_image, last_image]
        self.images.sort()
        self.first_image = self.images[0]
        self.last_image = self.images[1]
        self.analysed = False
    
    def combined_names(self):
        first = Path(self.first_image).stem
        last =  Path(self.last_image).stem
        fileout = "stacked_images"
        return fileout
    
    def get_output_folder_name(self):
        first = Path(self.first_image).stem
        last =  Path(self.last_image).stem
        now = datetime.today()
        dt_string = now.strftime("%Y%m%d_%H%M%S")
        folder_out = "{0}_{1}_pred_date_{2}".format(first, last, dt_string)
        return folder_out


class match_list(object):

    def __init__(self):
        self.list = []
        
    def __isEqual(self, a, b):
        for i, j in zip(a, b):
            if i != j:
                return False
        return True

    def append(self, item):
        for it in self.list:
            if self.__isEqual(it.images, item.images):
                return
        self.list.append(item)

    def item(self, idx):
        return self.list[idx]
    
    def size(self):
        return len(self.list)

def pandas_dict_to_list(df, field_name):
	return [df[field_name][i] for i in range(len(df[field_name]))]



def get_files_in_folder(folder, str_glob = '*.h5'):
		
	h5_files = [] 
	for path in Path(folder).rglob(str_glob):
		h5_files.append(path)
	return h5_files


class Folder_analysis(ABC):

	@abstractmethod
	def serialize(self):
		pass

	@abstractmethod
	def get_matches(self):
		pass



class Iceye_folder_analysis(Folder_analysis):

	def __init__(self, config):
	
		self.config = config
		self.extensions = {'h5': ['.h5', '.H5'], 'tif': ['.tif','.tiff', '.TIF', '.TIFF']}
		self.in_rootFolder  = config.CMD_LINE['data_path']
		self.out_rootFolder = config.CMD_LINE['root_output']
		self.partial_folder = os.path.join(self.out_rootFolder, "partial_iceye")
		self.serialized_fname = os.path.join(self.out_rootFolder, config.OUTPUT['serial_file'])		
		self.match_list = match_list()
		self.data_dict = {}
		self.__update_csv_file()

	def __get_files_in_folder(self):
		return list(Path(self.in_rootFolder).rglob('*.h5'))

	def __h5_to_tif(self, h5_name, tif_name, replace = False):

		if (not os.path.exists(tif_name) or replace):
			snap_processing.iceye.apply_snap_basic_process(h5_name, self.config, os.path.splitext(tif_name)[0])

	def __load_class_data(self):
		if (os.path.exists(self.serialized_fname)):
			self.__deserialize()

	def __deserialize(self):
		try:
			with open(self.serialized_fname, 'rb') as f:
				tmp_obj = pickle.load(f)
				self.data_dict = tmp_obj.data_dict
				self.match_list = tmp_obj.match_list
		except:
			print("Deserialization has failed")

	def __serialize(self):
		try:
			with open(self.serialized_fname, 'wb') as f:
				pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
		except:
			print("Serialization has failed")

	def __add_intersects(self, file, intersects):
		m = matches(file, intersects)
		self.match_list.append(m)

	def __update_csv_file(self):
		"""_summary_ called in the constructor
		"""
		self.__load_class_data()
		files = self.__get_files_in_folder()

		for f in files:
			hash_md5 = md5(f)
			if not self.data_dict.__contains__(hash_md5):
				d_dict = {"h5": None, "processed": None, "footprint": None}
				d_dict["h5"] = f
				d_dict["footprint"] = calculate_h5_footprint(f)
				#d_dict["processed"] = Path(self.out_rootFolder) / Path(self.partial_folder) / f.stem
				self.__h5_to_tif(f, d_dict["processed"])
				self.data_dict.update({hash_md5: d_dict})

		for di, dk in itertools.product(self.data_dict, self.data_dict): 
			if (di == dk):
				continue

			intersecs = find_tiff_intersections(self.data_dict[di]['footprint'], self.data_dict[dk]['footprint'])
			if intersecs:
				self.__add_intersects(self.data_dict[di]['processed'], self.data_dict[dk]['processed'])

		self.__serialize()


	def serialize(self):
		self.__serialize()

	def get_matches(self):
		return self.match_list



"""Folder analysis class for sentinel data"""
class Sentinel_folder_analysis(Folder_analysis):


	def __init__(self, config) -> None:
		super().__init__()
		self.extensions = "*.zip"
		self.in_rootFolder  = config.CMD_LINE['data_path']
		self.out_rootFolder = config.CMD_LINE['root_output']
		self.partial_folder = os.path.join(config.CMD_LINE['root_output'], "partial_sentinel")
		self.serialized_fname = os.path.join(config.CMD_LINE['root_output'], config.OUTPUT['serial_file'])
		self.config = config
		self.match_list = match_list()
		self.data_dict = {}
		self.update_csv_file()
	
	def apply_graph_step1(self, in_fname):
		ouf_folder = Path(self.out_rootFolder) / self.partial_folder / in_fname.stem

		# processed zip files are separated by folder so the precise filename is not required
		polygon = snap_processing.sentinel.apply_graph_step1(in_fname, ouf_folder, self.config) 

		return polygon

	def update_csv_file(self):

		self.__load_class_data()
		files = list(Path(self.in_rootFolder).rglob(self.extensions))
		assert len(files) > 0, f"No {self.extensions} file found in {self.in_rootFolder}"

		print(f"AppCensipam Info ~~~~ ({len(files)}) zip files have been found and will be processed using SNAP")
		
		with open( Path(self.out_rootFolder) / 'exec_report.txt', 'w') as report:
			report.write(f'#~~~~~~ Execution report #~~~~~~\n')
			report.write(f'Zip files found: {len(files)}\n')
			report.write(f'-----------------------------------------\n')

		for f in files:
			hash_md5 = md5(f)
			if not self.data_dict.__contains__(hash_md5):
				d_dict = {"h5": None, "processed": None, "footprint": None}
				d_dict["h5"] = f
				d_dict["footprint"] = self.apply_graph_step1(f)
				#d_dict["processed"] = Path(self.out_rootFolder) / self.partial_folder / f.stem 
				self.data_dict.update({hash_md5: d_dict})

		for di, dk in itertools.product(self.data_dict, self.data_dict): 
			if (di == dk):
				continue

			iou = self.data_dict[di]['footprint'].intersection(self.data_dict[dk]['footprint']).area / \
			self.data_dict[di]['footprint'].union(self.data_dict[dk]['footprint']).area
			intersecs = iou > 0.9

			if intersecs:
				#self.__add_intersects(self.data_dict[di]['processed'], self.data_dict[dk]['processed'])
				self.__add_intersects(self.data_dict[di]['h5'], self.data_dict[dk]['h5'])

		self.__serialize()

	def __load_class_data(self):
		if (os.path.exists(self.serialized_fname)):
			self.__deserialize()

	def __deserialize(self):
		try:
			with open(self.serialized_fname, 'rb') as f:
				tmp_obj = pickle.load(f)
				self.data_dict = tmp_obj.data_dict
				self.match_list = tmp_obj.match_list
		except:
			print("Deserialization has failed")

	def __serialize(self):
		try:
			with open(self.serialized_fname, 'wb') as f:
				pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
		except:
			print("Serialization has failed")

	def serialize(self):
		self.__serialize()


	def __add_intersects(self, file, intersects):
		m = matches(file, intersects)
		self.match_list.append(m)

	def get_matches(self):
		self.__write_report()
		return self.match_list

	def __write_report(self):

		with open( Path(self.out_rootFolder) / 'exec_report.txt', 'a') as report:
			report.write(f'#### Found matches\n')
			for it in self.match_list.list:	
				report.write(f'Match: {it.first_image.stem} {it.last_image.stem}\n')
			report.write(f'-----------------------------------------\n')


def create_folder_analysis(config) -> Folder_analysis:

	sentinel = Sentinel_folder_analysis
	iceye = Iceye_folder_analysis

	folder_class = eval(config.IN_DATA['format'])
	folder_obj = folder_class(config)

	return folder_obj


