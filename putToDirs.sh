#!/bin/bash

truthPath=$1
digitsPath=$2

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
		targetdir=${number:j:1}
		mv -- "$digitname" $targetdir
	done
done
