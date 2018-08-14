#!/bin/bash

if [[ $# != 3 ]] && [[ $# != 5 ]]; then
	echo 'Usage: ./putToDirs <ground_truths_txt_file> <digits_image_files_dir> <target_dir> [<start_idx> <end_idx>]'
	echo '<start_idx> and <end_idx> are optional and the default is inferred from ground truths file'
	exit -1
fi

truthPath="$1"
digitsPath="$2"
targetPath="$3"

readarray numbers < $truthPath

totallines="${#numbers[@]}"

start_idx=${4:-1}
end_idx=${5:-$totallines}


for i in {0..9}; do
	if ! [[ -d $i ]]; then
		mkdir "$targetPath/$i"
	fi
done

for (( i=start_idx; i<=end_idx; i++ )); do
	for (( j=0; j<6; j++ )); do
		digitname="$digitsPath/digit_${i}_$j.jpeg"
		number="${numbers[$((i-1))]}"
		targetdir="$targetPath/${number:j:1}"
		mv -- "$digitname" "$targetdir"
	done
done
