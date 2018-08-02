#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

inline Mat calcHist(const Mat& image);
int whichPattern(const Mat& image);

int main()
{
	for(int i = 1; i <= 100; i++){
		String file = "./data/t" + to_string(i) + ".png";
		Mat srcImage = imread(file, 0);
		imshow("Source Image", srcImage);
		
		int pattern = whichPattern(srcImage); // determine pattern of the image 

		Mat threshImg;
		int thresh;

		if(pattern == 1) // first pattern
			thresh = 220;
		else // second pattern
			thresh = 210;

		threshold(srcImage, threshImg, thresh, 255, THRESH_BINARY_INV);
		imshow("Thresh Image", threshImg);
		
		Mat labels, stats, centroids;
		int noComp = connectedComponentsWithStats(threshImg, labels, stats, centroids);
		
		Mat binaryImage(labels.size(), CV_8UC1);

		// holds if component will be included to foreground
		vector<bool> includeComp(noComp);
		vector<int> foreComps;
		includeComp[0] = false; // background
	
		for(int i = 1; i < noComp; i++)
			// if size less than <t> pixels dont include
			includeComp[i] = (stats.at<int>(i, CC_STAT_AREA) > 16) ? (foreComps.push_back(i), true) : false;
		

		for(int y = 0; y < labels.rows; y++)
			for(int x = 0; x < labels.cols; x++)
			{
				int label = labels.at<int>(y, x);
				if(includeComp[label] == true)
					binaryImage.at<uchar>(y, x) = 255;
				else
					binaryImage.at<uchar>(y, x) = 0;
			}
		
		
		int minCol = 30;
		
		int maxCol = 0;
		for(int i = 0; i < foreComps.size(); i++){
			int col = stats.at<int>(foreComps[i], CC_STAT_LEFT) + stats.at<int>(foreComps[i], CC_STAT_WIDTH);
			if(col > maxCol && col < 125)
				maxCol = col;
		}

		if(pattern == 2) // apply some offset for pattern2
			if(maxCol <= 111)
				maxCol = 111;
			else if (maxCol < 117)
				maxCol = 117;
		
		int minRow = binaryImage.rows;
		for(int i = 0; i < foreComps.size(); i++){
			int row = stats.at<int>(foreComps[i], CC_STAT_TOP);
			if(row < minRow )
				minRow = row;
		}

		int maxRow = 0;
		for(int i = 0; i < foreComps.size(); i++){
			int row = stats.at<int>(foreComps[i], CC_STAT_TOP) + stats.at<int>(foreComps[i], CC_STAT_HEIGHT);
			if(row > maxRow )
				maxRow = row;
		}

		Rect subReg = Rect(minCol, minRow, maxCol - minCol, maxRow - minRow);
		Mat subImage = Mat(binaryImage,subReg);
		
		vector<Mat> digits;

		Rect subImage1 = Rect(0, 0, subImage.cols / 2, subImage.rows);
		Rect subImage2 = Rect(subImage.cols / 2, 0, subImage.cols / 2, subImage.rows);

		Mat subMat1 = Mat(subImage, subImage1);
		Mat subMat2 = Mat(subImage, subImage2);

		int colIncrement1 = subMat1.cols / 3; 
		int colIncrement2 = subMat2.cols / 3; 
		int col = 0;
		for(int i = 0; i < 2; i++){
			Rect subDigit = Rect(col, 0, colIncrement1, subImage.rows);
			digits.push_back(Mat(subMat1, subDigit));
			col += colIncrement1;
		}
		Rect subDigit = Rect(col, 0, subMat1.cols-col, subImage.rows);
		digits.push_back(Mat(subMat1, subDigit));
		
		col = 0;
		for(int i = 0; i < 2; i++){
			Rect subDigit = Rect(col, 0, colIncrement2, subImage.rows);
			digits.push_back(Mat(subMat2, subDigit));
			col += colIncrement2;
		}
		subDigit = Rect(col, 0, subMat2.cols-col, subImage.rows);
		digits.push_back(Mat(subMat2, subDigit));

		for(int i = 0; i < digits.size(); i++){
			String name = "Digit" + to_string(i);
			imshow(name, digits[i]);
		}

		imshow("Sub Image", subImage);
		//imshow("RESULT", binaryImage);
	
		waitKey();
	}
}

Mat calcHist(const Mat& image)
{

	int histSize = 256;
	float range[] = {0, 256};
	const float* histRange = {range};
	int channels[] = {0};
	Mat imgHist; 

	calcHist(&image, 1, channels, Mat(), imgHist, 1, &histSize, &histRange);

#if 0
	int distinctIntensityCount = 0;
	for(int i = 0; i < histSize; i++)
		if(imgHist.at<float>(i) != 0){
			cout << i << ": " << imgHist.at<float>(i) << endl;
			distinctIntensityCount++;
		}

	cout << distinctIntensityCount << endl;
#endif
	
	return imgHist;
}

// finds out whether given image falls under pattern1 or pattern2
int whichPattern(const Mat& image)
{
	Mat hist = calcHist(image);
	
	// pattern1 images has almost more than 600 pixels
	// for 0(pure black) intensity, lets set threshold to 500
	int thresh = 500;
	if(hist.at<float>(0) > thresh)
		return 1; // pattern1
	else
		return 2; // pattern2

	
}
