#!/bin/bash

hash_file="cur_HASHFILE.txt"

function show_box()
{
	SHOW_BOX_MSG=$1
	SHOW_BOX_L=${#SHOW_BOX_MSG}
	SHOW_BOX_L=$((SHOW_BOX_L+4))
	SHOW_BOX_LINE=""
	for SHOW_BOX_i in $(seq $SHOW_BOX_L)
	do
		SHOW_BOX_LINE=${SHOW_BOX_LINE}"-"
	done
	echo
	echo ${SHOW_BOX_LINE}
	echo \| $SHOW_BOX_MSG \|
	echo ${SHOW_BOX_LINE}
	echo
}


########################################

show_box "Teste na pasta atual"

rm -f "arq*.txt"

for i in {1..5}; do
	echo File \# ${i} > "arq ${i}.txt"
done

./get_folder_hashes.sh . "${hash_file}"

for i in {6..9}; do
	echo File \# ${i} > "arq ${i}.txt"
done

./get_folder_changes.sh . "${hash_file}"

rm arq*.txt ${hash_file}

########################################

show_box "Teste na pasta criada \"arqs_teste\""

FOLDER="arqs_teste"
rm -rf "${FOLDER}"
mkdir "${FOLDER}"

for i in {1..5}; do
	echo File \# ${i} > "${FOLDER}/arq ${i}.txt"
done

./get_folder_hashes.sh "${FOLDER}" "${FOLDER}/${hash_file}"

for i in {6..9}; do
	echo File \# ${i} > "${FOLDER}/arq ${i}.txt"
done

./get_folder_changes.sh "${FOLDER}" "${FOLDER}/${hash_file}"

rm -rf ${FOLDER}

########################################

show_box "Teste na pasta criada \"arqs teste\""

FOLDER="arqs teste"
rm -rf "${FOLDER}"
mkdir "${FOLDER}"

for i in {1..5}; do
	echo File \# ${i} > "${FOLDER}/arq ${i}.txt"
done

./get_folder_hashes.sh "${FOLDER}" "${FOLDER}/${hash_file}"

for i in {6..9}; do
	echo File \# ${i} > "${FOLDER}/arq ${i}.txt"
done

./get_folder_changes.sh "${FOLDER}" "${FOLDER}/${hash_file}"

rm -rf "${FOLDER}"