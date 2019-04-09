#!/usr/bin/python
import sys
import cv2
import torch
from utils.divDigits import separate_image

model = torch.load('model.pt', map_location='cpu')

def break_captcha(image):
    digits = [[cv2.resize(img, (14, 24))] for img in separate_image(image)]
    tens = torch.tensor(digits, dtype=torch.float32) / 255
    with torch.no_grad():
        result = model(tens)
        _, nums = torch.max(result, 1)

    out = [num.item() for num in nums]

    captcha = ''.join([str(i) for i in out])
    return captcha

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python breaker.py <image_file1> [<image_file2> ...]')
        sys.exit(-1)

    for i in sys.argv[1:]:
        print(break_captcha(i))
