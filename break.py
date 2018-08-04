#!/usr/bin/python
import sys
import cv2
import torch
from divDigits import separate_image

model = None

def break_captcha(image):
	digits = [[cv2.resize(img, (14, 24))] for img in separate_image(image)]
	tens = torch.tensor(digits, dtype=torch.float32).cuda() / 255
	with torch.no_grad():
		result = model(tens)
		_, nums = torch.max(result, 1)

	out = [num.item() for num in nums]

	captcha = [out[0], out[2], out[4], out[1], out[3], out[5]]
	captcha = ''.join([str(i) for i in captcha])
	print(captcha)

if __name__ == '__main__':
	if len(sys.argv) < 2:
		print('Usage: python break.py <image_file1> [image_file2 ...]')
		sys.exit(-1)

	model = torch.load('model.pt')
	for i in sys.argv[1:]:
		break_captcha(i)
