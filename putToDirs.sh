#!/bin/bash

if [[ $# != 3 ]]; then
	echo 'Usage: ./putToDirs <ground_truths_txt_file> <digits_image_files_dir> <target_dir>'
	exit -1
fi

truthPath="$1"
digitsPath="$2"
targetPath="$3"

readarray numbers < $truthPath

totallines="${#numbers[@]}"

for i in {0..9}; do
	if ! [[ -d $i ]]; then
		mkdir $i
	fi
done

for (( i=1; i<=totallines; i++ )); do
	for (( j=0; j<6; j++ )); do
		digitname="$digitsPath/digit_${i}_$j.jpeg"
		number="${numbers[$((i-1))]}"
		targetdir="$targetPath/${number:j:1}"
		mv -- "$digitname" "$targetdir"
	done
done
