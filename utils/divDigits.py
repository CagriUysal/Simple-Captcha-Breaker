import cv2
import sys
import numpy as np

#obtained through experiment 
pattern1_thresh = 170
pattern2_thresh = 155
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

    # use median blue for pattern1 and gaussian blur for patter2 images as pre-processing
    srcImage = cv2.medianBlur(srcImage, 3) if pattern == 1 else cv2.GaussianBlur(srcImage, (3,3), 0)
    
    ret, threshImage = cv2.threshold(srcImage, thresh, 255, cv2.THRESH_BINARY_INV)

    # get connected components
    numLabel, labelImage, stats, centroids = cv2.connectedComponentsWithStats(threshImage, 8, cv2.CV_32S)
    
    # holds if compenet will be included to foreground
    foreComps = [i for i in range(1, numLabel) if stats[i, cv2.CC_STAT_AREA] >= min_connected_len]
    
    
    minCol = 30; # seen that all digits start at 30th column 
                             # no need for additional computation
    
    
    # Get binary image after erasing some connected components those areas under the threshold
    binaryImage = np.zeros_like(srcImage)
    labelImage = np.array(labelImage)
    for k in [np.where(labelImage == i) for i in foreComps]:
        binaryImage[k] = 255

    # find the boundaries where digits present in the image 
    array = np.array([stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH]  for i in foreComps])
    maxCol = max(array[np.where(array < 125)]) # observed that digits right boundary never exceeds 125th pixel 
                                                                                         # thus this one prevents false boundaries
    
    # find boundaries in y axis
    minRow = min([stats[i, cv2.CC_STAT_TOP] for i in foreComps])
    maxRow = max([stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] for i in foreComps])

    
    subImage = threshImage[minRow:maxRow, minCol:maxCol]
    
    # Sub image divided to half in order to segment digit's more precisely
    subImage1 = subImage[:, :int(subImage.shape[1]/2)]
    subImage2 = subImage[:, int(subImage.shape[1]/2):]

    colIncrement1 = subImage1.shape[1] / 3
    colIncrement2 = subImage2.shape[1] / 3
    
    # get segmented digits as list
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
    digitList = digitList1 + digitList2
    
    # It is observed that sometimes part of the previous digit shifted to 
    # digit that follows it.
    # Following prevents it most of the time
    for i in range(1, 6):
        # Appy slight closing before finding c.c, sometimes particular digit parts only apart by 1 pixels (fill them to avoid false shifting)
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
        closedDigit= cv2.morphologyEx(digitList[i], cv2.MORPH_CLOSE, kernel, 1)

        numLabel, labelImage, stats, centroids = cv2.connectedComponentsWithStats(closedDigit, 8, cv2.CV_32S)
        #print(numLabel)
        if numLabel > 2: # consider shifting if there is more than 1(with background its 2)  c.c. per digit
            for j in range(1,numLabel):
                # If it touches left border of the image
                # and it's width doesnt exceed 35 percent of the digit, assume it is a shifting
                if stats[j, cv2.CC_STAT_LEFT] == 0 and stats[j, cv2.CC_STAT_LEFT] + stats[j, cv2.CC_STAT_WIDTH] < 0.35 * digitList[i].shape[1]:
                    # find the part that will attach to previous digit
                    startCol = stats[j, cv2.CC_STAT_LEFT]
                    width_shifted = stats[j, cv2.CC_STAT_WIDTH]
                    shiftedPart = digitList[i][:, startCol:startCol+width_shifted]  

                    # shift previous image to left while maintain its size 
                    # shift amount is width of the shifted-part
                    fixedDigit = np.zeros_like(digitList[i - 1])
                    fixedDigit[:, 0:fixedDigit.shape[1] - width_shifted] = digitList[i - 1][:, width_shifted:fixedDigit.shape[1]]
          
                    # appending shifted part back to previous digit
                    fixedDigit[:, fixedDigit.shape[1]-width_shifted:fixedDigit.shape[1]] = shiftedPart[:]
                        
                    digitList[i - 1] = fixedDigit # replace digit with fixed digit
                    
                    # remove shifted part from current digit 
                    digitList[i][:, 0:width_shifted] = 0
                     

    return digitList
  
def divDigits(args):
    if(len(args) != 5):
        print("Usage: python divDigits.py <path-to-data> <digit-write-path> <start-index> <end-index>")
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
