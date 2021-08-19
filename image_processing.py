'''
References: https://pythonspot.com/nltk-stop-words/
            https://docs.python.org/3/library/re.html
            https://scikit-learn.org/stable/modules/generated/
            sklearn.naive_bayes.MultinomialNB.html
            https://scikit-learn.org/stable/modules/generated/
            sklearn.feature_extraction.text.CountVectorizer.html
            https://scikit-learn.org/stable/modules/generated/
            sklearn.feature_extraction.text.TfidfTransformer.html
Assumptions:
            1. The following images are present in the same directory as
            the python file:
                avengers_imdb.jpg
                bush_house_wikipedia.jpg
                forestry_commission_gov_uk.jpg
                rolland_garros_tv5monde.jpg
            2. An 'outputs' folder will already be created in the current
            directory where the code and image files are kept
'''
import skimage.io
import skimage.util as util
import skimage.filters as sk_filter
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from scipy import ndimage, misc
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
import skimage.feature as feat
import skimage.transform as transform
from skimage.filters import threshold_mean
import numpy as np
from datetime import datetime

print(datetime.now())
# Question 1
# read image
original = skimage.io.imread(fname="avengers_imdb.jpg")

# Determine the size of the avengers imdb.jpg image
print('Size of the avengers_imdb.jpg: '+str(original.size))

print('Shape of the avengers_imdb.jpg: '+str(original.shape))

# Convert the image to grayscale
grayscale = rgb2gray(original)

# Convert the image to black and white
# Fetch the mean value of the pixel intensities
thresh = threshold_mean(grayscale)
# Apply the threshold value to compute the binary pixel
bw_img = grayscale > thresh

# Plot and save the images
fig, axes = plt.subplots(1, 3, figsize=(8, 4))
ax = axes.ravel()

ax[0].imshow(original)
ax[0].set_title("Original")
ax[1].imshow(grayscale, cmap=plt.cm.gray)
ax[1].set_title("Grayscale")
ax[2].imshow(bw_img, cmap=plt.cm.gray)
ax[2].set_title("Black and White")

fig.tight_layout()
# plt.show()
plt.savefig('outputs/ImageProcessing_Ques1.png')

# Question 2
# Read image
image = skimage.io.imread(fname="bush_house_wikipedia.jpg")

# Add Gaussian random noise
new_image = util.random_noise(image,'gaussian',var=0.1)

# Filter the perturbed image with Gaussian Mask
new_filter_image = sk_filter.gaussian(new_image,sigma=1,multichannel=False)
# new_filter_image = gaussian_filter(new_image, sigma=1)
fig, axes = plt.subplots(1, 4, figsize=(8, 4))
ax = axes.ravel()

# Filter the image with uniform smoothing mask of size 9x9
filter_image_unif = ndimage.uniform_filter(new_image,size=9)

ax[0].imshow(image)
ax[0].set_title("Original")
ax[1].imshow(new_image)
ax[1].set_title("Random Noise")
ax[2].imshow(new_filter_image)
ax[2].set_title("Filter Gaussian Mask")
ax[3].imshow(filter_image_unif)
ax[3].set_title("Filter Uniform Mask")

fig.tight_layout()
# plt.show()
plt.savefig('outputs/ImageProcessing_Ques2.png')

# Question 3

# Read image
image = skimage.io.imread(fname="forestry_commission_gov_uk.jpg")

# Divide image into 5 segments using k-means segmentation
segments = slic(image, n_segments=5, compactness=20,start_label=1)


# Plot and save the images
fig, axes = plt.subplots(1, 2, figsize=(8, 4))
ax = axes.ravel()

ax[0].imshow(image)
ax[0].set_title("Original")
ax[1].imshow(mark_boundaries(image, segments))
ax[1].set_title("Segmented")

fig.tight_layout()
# plt.show()
plt.savefig('outputs/ImageProcessing_Ques3.png')

# Question 4
# Perform Canny edge detection and apply Hough transform on
# rolland_garros_tv5monde.jpg

# Read image
image = skimage.io.imread(fname="rolland_garros_tv5monde.jpg")

# Convert image to grayscale
grayscale = rgb2gray(image)

# Compute canny filter
edges = feat.canny(grayscale)

# Plot and save the images
fig, axes = plt.subplots(1, 3, figsize=(10, 4))
ax = axes.ravel()

# Hough Transform
lines = transform.probabilistic_hough_line(edges,threshold=10,line_length=10,
                                           line_gap=3)

ax[0].imshow(image)
ax[0].set_title("Original")
ax[1].imshow(edges , cmap=plt.cm.gray)
ax[1].set_title("Canny Edge")
ax[2].imshow(edges * 0)
for line in lines:
    p0, p1 = line
    ax[2].plot((p0[0], p1[0]), (p0[1], p1[1]))
ax[2].set_title('Probabilistic Hough Transform')

fig.tight_layout()
plt.savefig('outputs/ImageProcessing_Ques4.png')

print(datetime.now())
