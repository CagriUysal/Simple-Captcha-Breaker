import cv2
import sys
import numpy as np

#obtained through experiment 
pattern1_thresh = 170
pattern2_thresh = 145
min_connected_len = 15

def whichPattern(image):
	hist = cv2.calcHist([image], [0], None, [256], [0, 256])
	# pattern1 images has almost more than 600 pixels
	# for 0(pure black) intensity, lets set threshold to 500
	thresh = 500	
	if(hist[0, 0] > thresh):
		return 1
	else:
		return 2

# Separates the image <file> to six subimages containing each digit
# With an attempt to internally eliminate noise and color inversion
def separate_image(file):
	srcImage = cv2.imread(file, 0)

	pattern = whichPattern(srcImage) # determine the pattern of the image

	thresh = pattern1_thresh if pattern == 1 else pattern2_thresh

	# test if gaussian works
	srcImage = cv2.medianBlur(srcImage, 3) if pattern == 1 else cv2.GaussianBlur(srcImage, (3,3), 0)
	
	#if pattern == 1:
	#	srcImage2 = cv2.GaussianBlur(srcImage, (3,3), 0)

	#cv2.imshow("Median", srcImage)
	
	#global threshImage
	#while cv2.waitKey() != ord('c') and pattern1_thresh != 255:
	ret, threshImage = cv2.threshold(srcImage, thresh, 255, cv2.THRESH_BINARY_INV)
	#cv2.imshow("threshold", threshImage)
	#print(thresh, end='\r', flush=True)
	#thresh += 1
	'''
	if pattern == 1:	
		kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
		#threshImage = cv2.morphologyEx(threshImage, cv2.MORPH_DILATE, kernel, 1)
		#cv2.imshow("after dilate", threshImage)
		threshImage = cv2.morphologyEx(threshImage, cv2.MORPH_CLOSE, kernel, 2)
		cv2.imshow("after close", threshImage)
		#kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
		#threshImage = cv2.morphologyEx(threshImage, cv2.MORPH_CLOSE, kernel, 1)
		#cv2.imshow("after close", threshImage)
		#threshImage = cv2.morphologyEx(threshImage, cv2.MORPH_ERODE, kernel, 1)
		#cv2.imshow("after erode", threshImage)
	'''	
	numLabel, labelImage, stats, centroids = cv2.connectedComponentsWithStats(threshImage, 8, cv2.CV_32S)
	
	# holds if compenet will be included to foreground
	foreComps = [i for i in range(1, numLabel) if stats[i, cv2.CC_STAT_AREA] >= min_connected_len]
	
	
	minCol = 30; # seen that all digits start at 30th column 
							 # no need for additional computation
	
	binaryImage = np.zeros_like(srcImage)
	labelImage = np.array(labelImage)
	for k in [np.where(labelImage == i) for i in foreComps]:
		binaryImage[k] = 255
	

	array = np.array([stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH]  for i in foreComps])
	maxCol = max(array[np.where(array < 125)])
		
	# Dont touch
	if pattern == 2:
		if maxCol <= 111:
			axCol = 111
		elif maxCol < 117:
			maxCol = 117

	minRow = min([stats[i, cv2.CC_STAT_TOP] for i in foreComps])
	maxRow = max([stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] for i in foreComps])

	#subImage = binaryImage[minRow:maxRow, minCol:maxCol]
	#cv2.imshow("subImage", subImage)
	
	subImage = threshImage[minRow:maxRow, minCol:maxCol]
	#cv2.imshow("subImage", subImage1)

	subImage1 = subImage[:, :int(subImage.shape[1]/2)]
	subImage2 = subImage[:, int(subImage.shape[1]/2):]

	colIncrement1 = subImage1.shape[1] / 3
	colIncrement2 = subImage2.shape[1] / 3
	
	digitList1 = []
	digitList2 = []
	
	col1 = 0
	col2 = 0
	for i in range(2):
		digitList1.append(subImage1[:, int(col1):int(col1+colIncrement1)])
		digitList2.append(subImage2[:, int(col2):int(col2+colIncrement2)])
		col1 += colIncrement1
		col2 += colIncrement2

	digitList1.append(subImage1[:, int(col1):])
	digitList2.append(subImage2[:, int(col2):])
	
	if cv2.waitKey() == ord('q'):
		sys.exit(1)

	return digitList1 + digitList2
  
def divDigits(args):
	if(len(args) != 5):
		print("Usage: breaker <path-to-data> <digit-write-path> <start-index> <end-index>")
		sys.exit(-1)

	path_to_data = args[1];
	digit_write_path = args[2];
	start_index = int(args[3]);
	end_index = int(args[4]);

	for idx in range(start_index, end_index + 1):
		file = path_to_data + "/" + str(idx) + ".jpeg"
		digit_list = separate_image(file)
		print(idx, end='\r', flush=True)
		for j in range(6):
			cv2.imwrite(digit_write_path + '/digit_' + str(idx) + '_' +  str(j) + '.jpeg', digit_list[j])

if __name__ == '__main__':
	divDigits(sys.argv)
