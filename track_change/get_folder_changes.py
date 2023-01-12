import os

######################################
# Código abaixo funciona com pastas  #
# que contenham espaços no nome, mas #
# usa scripts Linux                  #
######################################

def get_folder_hashes(folder_path, output_hashfile_name):
	# os.system("./get_folder_hashes.sh '" + folder_path + \
	# 	"' '" + output_hashfile_name + "'")
	os.system("./track_change/get_folder_hashes.sh '" + folder_path + \
		"' '" + output_hashfile_name + "'")

def get_folder_changes(folder_path, input_hashfile_name):
	return os.popen("./track_change/get_folder_changes.sh '" + \
		folder_path + "' '" + input_hashfile_name + "'").read().splitlines()

#########################################
# Código abaixo não funciona com pastas #
# que contenham espaços no nome         #
#########################################

# import hashlib

# def md5(fname):
# 	hash_md5 = hashlib.md5()
# 	with open(fname, "rb") as f:
# 		for chunk in iter(lambda: f.read(4096), b""):
# 			hash_md5.update(chunk)
# 	return hash_md5.hexdigest()

# def get_folder_hashes(folder_path, output_hashfile_name):
# 	if output_hashfile_name.find("HASHFILE")==-1:
# 		print("Hash file name '" + output_hashfile_name + \
# 			"' must contain the word 'HASHFILE'.")
# 		return
# 	os.system("rm -rf '" + output_hashfile_name + "'")
# 	folder_files = os.listdir(folder_path) # next(os.walk(folder_path),(None, None, []))[2]
# 	# print("-"*100)
# 	# print(folder_files)
# 	# print("-"*100)
# 	with open(output_hashfile_name.encode('unicode_escape'), 'w') as fp:
# 		for cur_file in folder_files:
# 			if cur_file.find("HASHFILE")==-1 and os.path.isfile(cur_file):
# 				fp.write(folder_path + "/" + cur_file + \
# 					"  " + md5(folder_path + '/' + cur_file) + "\n")
# 	os.system("sort '" + output_hashfile_name + "' -o " + \
# 		output_hashfile_name)

# def get_folder_changes(folder_path, input_hashfile_name):
# 	new_hashfile  = input_hashfile_name + "_NEW.txt"
# 	diff_hashfile  = input_hashfile_name + "_DIFF.txt"
# 	get_folder_hashes(folder_path, new_hashfile)
# 	diff_system_out = os.popen("diff " + new_hashfile + " " + \
# 		input_hashfile_name + \
# 		' | sed "s/^< //" | sed "s/  ................................$//" | tail -n +2').read()
# 	os.system("rm -rf " + new_hashfile)
# 	return diff_system_out.splitlines()
