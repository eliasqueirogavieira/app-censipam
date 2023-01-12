import os
import hashlib

def get_folder_hashes(folder_path):
	hashes = []
	for cur_file in os.listdir(folder_path):
		if folder_path[-1]=='/':
			cur_folder_file = folder_path + cur_file
		else:
			cur_folder_file = folder_path + "/" + cur_file
		if os.path.isfile(cur_folder_file):
			hashes.append({"file_name": cur_folder_file,
				"md5": md5(cur_folder_file)})
	return sorted(hashes, key=lambda d: d['file_name']) 

def get_folder_changes(folder_path, folder_hashes):
	new_hashes = get_folder_hashes(folder_path)
	return [new_hash_items for new_hash_items in new_hashes 
		if new_hash_items not in folder_hashes]
	
def md5(fname):
	hash_md5 = hashlib.md5()
	with open(fname, "rb") as f:
		for chunk in iter(lambda: f.read(4096), b""):
			hash_md5.update(chunk)
	return hash_md5.hexdigest()