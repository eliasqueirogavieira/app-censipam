#!/bin/bash

if [[ $# -eq 0 || "$1" == "-h" || "$1" == "--help" ]]; then
	echo
	echo "      "$0 \"FOLDER_NAME\" \"OUTPUT_HASHFILE_NAME\"
	echo
	echo "      Looks for changes in folder \"FOLDER_NAME\""
	echo "      based in old SHA1 checksums stored in the file \"OUTPUT_HASHFILE_NAME\"."
	echo
	echo "      New files are written displayed on screen, separated by newlines."
	echo 
	echo "      Use double quotation marks (\"\") to avoid problems"
	echo "      with folder and file names containing spaces."
	echo
	echo "      The file \"OUTPUT_HASHFILE_NAME\" is not updated at the end of this script."
	echo
	exit
fi

hash_file="${2}_NEW.txt" #"$(dirname "${2}")/NEW_$(basename "${2}")"
hash_file_diff="${2}_DIFF.txt" #"$(dirname "${2}")DIFF_$(basename "${2}")"

FOLDER="${1}"

./get_folder_hashes.sh "${FOLDER}" "${hash_file}"

diff "${hash_file}" "${2}" | sed "s/^< ........................................  //" > "${hash_file_diff}"

OIFS="$IFS"
IFS=$'\n'
for f in $(cat "${hash_file_diff}")
do
	if [ -f "${f}" ]; then
		echo "$f"
	fi
done
IFS="$OIFS"

# echo "......................................................"
# echo Hashfile "${hash_file}":
# cat "${hash_file}"
# echo "......................................................"
# echo Hash diff file "${hash_file_diff}":
# cat "${hash_file_diff}"
# echo "......................................................"

rm "${hash_file}" "${hash_file_diff}"