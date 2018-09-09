import cv2

path = "/home/murdoc/passData/dataset/"
digits_path = "/home/murdoc/passData/data-digits/v01/"
def main():
  index = 1
  while(True):
    keyPressed = cv2.waitKey(0)
    if keyPressed ==  ord('q'):
      break
    elif keyPressed == ord('n'):
      index += 1
    elif keyPressed == ord('b'):
      index -= 1
    elif keyPressed == ord('i'):
      index = int(input('enter index: '))

    print(index)
    org_path = path + str(index) + '.jpeg'
    org_img = cv2.imread(org_path)
    cv2.imshow('Original Image',org_img)
    for i in range(0,6):
      digit_path = digits_path + 'digit_' +str(index) + '_' + str(i) + '.jpeg'
      digit = cv2.imread(digit_path)
      cv2.imshow(str(i), digit)
         
if __name__== '__main__':
  main()
