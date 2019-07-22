# Simple-Captcha-Breaker
Simple captcha breaker is a fun project for solving (now obsolete) captchas used in METU's course registration website.
Images used in training and testing obtained via a web bot. There was two different captcha images, we called them type1 and type2.
We have achieved 99.1% accuracy with 4500 training images and 500 test images.</br>

<p float="left">
<img src="https://image.ibb.co/igkbJ0/resimler1.png" alt="drawing" width="400"/>
<img src="https://image.ibb.co/fiBNrL/resimler2.png" alt="drawing" width="400"/>
</p>

### [Tutorial of the project](https://cagriuysal.github.io/Simple-Captcha-Breaker/)

## Prerequisites
```
python
pytorch
opencv
numpy
```
## Training
Download dataset and ground truths provided.

* Captcha Dataset: [dataset](https://mega.nz/#!2Xo0lYxI!utIihUv511jwJXCti4g35yjhl9ogQxjc2sTkZ2BW-Aw) 
* Captcha Ground Truths: [GT](https://mega.nz/#!WOJ2SYiZ!kdbwDzo7MguFsuumRsYzQ58PI8yDsoAexkwQyJJdKdQ)

1. First unzip dataset and ground truths to a folder. (Dataset contains 5299 images and first 5000 image have ground truths.)
Then use saveDigits.py which separates digits in the images and puts them to distinct folders, i.e. 0,1,..,9 so that pytorch 
dataloader could infer the image classes.
```
python saveDigits.py <dataset-path> <train-digits-path> <startIndex> <endIndex> <groundTruth-path>
```
e.g. ```python saveDigits.py /home/user/dataset /home/user/train 1 4500 /home/user/labels.txt``` will use first 4500 images as train data and puts
4500 * 6 digits as jpeg files to appropriate folders. Different intervals can be used for train data.

2. ``` python train.py ```
By default `train.py` looks for a folder named `train` consisting of subfolders `0..9`. This is the folder we created in step 1.
You can edit folder path if you want.

There is a pre-trained [model](https://mega.nz/#!mWYl1IpT!o-sqt5oHZWxZu7jfz9lJg9FbCJWOnsU-E_jqXSOlnfY) as an example, remember to change file name to ```model.pt```. 

## Testing
Steps are the similar as training. Apply the unused image interval as test interval.

1. Again use saveDigits to separate digits to folders.
```
python saveDigits.py <dataset-path> <test-digits-path> <startIndex> <endIndex> <groundTruth-path>
```
e.g. ```python saveDigits.py /home/user/dataset /home/user/test 4501 5000 /home/user/labels.txt```  
Thus we used  [1, 4500] in training phase, remaining [4501, 5000] interval used as test images.

2. ```python test.py``` Again by default `test.py` looks for a folder named `test` consisting of subfolders `0..9`.  

## Individual Running
Trained model can be run on whole image rather than separated digits via breaker.py.
```
python breaker.py <image_file1> [<image_file2> ...]
```
e.g. ```python breaker.py /home/user/dataset/4585.jpeg``` gives the predicted number in the single image.  
or   ```python breaker.py /home/user/dataset/{4510..4560}.jpeg``` gives the set of predicted numbers.

## Authors
* [yalpul](https://github.com/yalpul)  
* [CagriUysal](https://github.com/CagriUysal)  
* [ArdaEs](https://github.com/ArdaEs)

