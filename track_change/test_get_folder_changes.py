from get_folder_changes import *
import os

hash_file = "cur_HASHFILE.txt"
folders = [".", "arqs_teste", "arqs teste"]
cur_dir = os.getcwd()

def create_dummy_file(folder, number):
	os.system(('echo File \\# %d > ' % number) + \
			"'" + folder + "'/" + ('"arq %d.txt"' % number))

for folder in folders:
	if folder[-1]=='/':
		folder = folder[0:-1]
	if folder!="./" and folder!='.':
		os.system("rm -rf '" + folder + "'")
		os.system("mkdir '" + folder + "'")
	else:
		os.system('rm -rf arq*.txt')
	print("########################################")
	print("Teste na pasta '" + folder + "'")
	print("Criando arquivos...")
	[create_dummy_file(folder, i) for i in range(5)]
	print("Calculando hashes...")
	get_folder_hashes(folder, folder + "/" + hash_file)
	print("Criando novos arquivos...")
	[create_dummy_file(folder, i) for i in range(5,10)]
	print("Conferindo lista de arquivos novos...")
	lista_arqs_novos = get_folder_changes(folder, folder + "/" + hash_file)
	print(lista_arqs_novos)
	if folder!="./" and folder!='.':
		os.system("rm -rf '" + folder + "'")
	else:
		os.system('rm -f arq*.txt ' + hash_file)

