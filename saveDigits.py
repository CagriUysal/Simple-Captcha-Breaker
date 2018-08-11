#!/usr/bin/python
import sys
import cv2
from divDigits import separate_image
import os

def save_digits(readPath, writePath, start, end, gT):
  
  # read ground truths 
  with open(gT, 'rU') as f:
    numbers = [[j for j in list(i)] for i in f.read().splitlines()]
  
	# create write path if necessary
	if not os.path.exists(writePath):
		os.mkdir(writePath)

  # check if dirs for individual digits exits
  # if not create {0,1 .. 9}
  for i in range(10):
    if not os.path.exists(writePath + '/' + str(i)):
      os.mkdir(writePath + '/' + str(i))
  
  number_ind = 0
  for i in range(start, end+1):
    image = readPath + '/' + str(i) + '.jpeg'
    digits = separate_image(image)
    digit_ind = 0
    for img in digits:
      cv2.imwrite(writePath + '/' + numbers[number_ind][digit_ind] + '/' + str(i) + '_' + str(digit_ind) + '.jpeg', img)
      digit_ind += 1
    number_ind += 1

if __name__ == '__main__':
	if len(sys.argv) != 6:
		print('Usage: ./saveDigits.py <readPath> <writePath> <startIndex> <endIndex> <groundTruths>')
		sys.exit(-1)

	save_digits(sys.argv[1],sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5])
  
