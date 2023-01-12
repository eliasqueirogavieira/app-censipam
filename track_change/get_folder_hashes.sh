#!/bin/bash

if [[ $# -eq 0 || "$1" == "-h" || "$1" == "--help" ]]; then
	echo
	echo "      "$0 \"FOLDER_NAME\" \"OUTPUT_HASHFILE_NAME\"
	echo
	echo "      Calculates SHA1 checksums for all files in folder \"FOLDER_NAME\","
	echo "      and outputs them to the file \"OUTPUT_HASHFILE_NAME\"."
	echo
	echo "      The hash file name \"OUTPUT_HASHFILE_NAME\" must contain"
	echo "      the word \"HASHFILE\". For example, \"cur_HASHFILE.txt\"."
	echo 
	echo "      Use double quotation marks (\"\") to avoid problems"
	echo "      with folder and file names containing spaces."
	echo
	exit
fi

hash_file="$2"

if [ ! "$(echo "${2}" | grep HASHFILE)" ]; then
	echo "Hash file name \"${hash_file}\" must contain the word \'HASHFILE\'."
	exit
fi

rm -f "${hash_file}"
touch "${hash_file}"

OIFS="$IFS"
IFS=$'\n'
for f in ${1}/*
do
	if [ ! $(echo "${f}" | grep HASHFILE) ]; then
		if [ -f "${f}" ]; then
			sha1sum "${f}" >> "${hash_file}"
		fi
	fi
done
IFS="$OIFS"
