# Image-Processing
Image Processing using Skimage
Versions:
python '3.8.5'
pandas '1.1.4'
sklearn '0.24.1'
matplotlib '3.3.2'
numpy '1.19.3'
skimage '0.18.1'
nltk '3.5'
re '2.2.1'
scipy '1.6.0'

Folder requirement: A folder named 'outputs' should be created in the directory where the python
		    and data files are kept. The result images will be saved in the outputs 
		    folder.
Files: The below files (python and data) should be placed in the same folder:
	image_processing.py
	avengers_imdb.jpg
	bush_house_wikipedia.jpg
	forestry_commission_gov_uk.jpg
	rolland_garros_tv5monde.jpg
  
  *******************************************************************************************************
*******************************************IMAGE PROCESSING********************************************
*******************************************************************************************************	

Files:
image_processing.py
avengers_imdb.jpg
bush_house_wikipedia.jpg
forestry_commission_gov_uk.jpg
rolland_garros_tv5monde.jpg

Code Structure:
1. a) The image avengers_imdb.jpg is read using skimage module
   b) Determine the size of the avengers imdb.jpg image using size() and shape() functions
   c) Convert the image to grayscale using the rgb2gray() function of the skimage.color module
   d) Convert the image to black and white by fetching the mean value of the pixel intensities of the
grayscale image and then the threshold value is used to compute the binary pixel
   e) The original and the converted images are plotted and saved in the outputs folder using matplotlib,
the name of the resulting image is 'ImageProcessing_Ques1.png'.
2. a)The image bush_house_wikipedia.jpg is read using skimage module
   b) Gaussian random noise is added to the original image using util.random_noise() function of the 
skimage module
   c) On the perturbed image, gaussian filter and uniform filter is separately applied using the gaussian()
function of the skimage.filters module and uniform_filter() function of the ndimage module respectively
   e) The original and the converted images are plotted and saved in the outputs folder using matplotlib,
the name of the resulting image is 'ImageProcessing_Ques2.png'.
3. a) The image forestry_commission_gov_uk.jpg is read using skimage module
   b) The K-Means segmentation has been performed by using the slic() function of the skimage.segmentation
module
   c) The original and the segmented images are plotted and saved in the outputs folder using matplotlib,
the name of the resulting image is 'ImageProcessing_Ques3.png'.
4. a) The image rolland_garros_tv5monde.jpg is read using skimage module
   b) The image has been converted to grayscale 
   c) The canny_edge() function of the skimage.feature module has been used to compute the canny edges of 
the original image. 
   d) Probabilistic Hough Transform is then performed on the edges of the original image using the function
probabilistic_hough_line() of skimage.transform module
   e) The original and the segmented images are plotted and saved in the outputs folder using matplotlib,
the name of the resulting image is 'ImageProcessing_Ques3.png'.



*******************************************************************************************************
***********************************************END*****************************************************
*******************************************************************************************************
