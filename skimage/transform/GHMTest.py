from GHM import GHM
from GHMHelperFuncs import *

import numpy as np
# import traceback
import matplotlib.pyplot as plt

# TODO test that image matched to itself does not change
# TODO test that image matched to a template that is all one value X changes to be completely X (black and white are example templates for this, we should probably also test a uniform gray image)

def create_black(dimensions=(100,100)):
	return np.zeros(dimensions)

def create_white(dimensions=(100,100)):
	return np.ones(dimensions)*255

def save_img(img, filename, folder="."):
	filepath = folder + "/" + filename
	np.save(filepath, img)
	plt.imsave(filepath + ".jpg", img, cmap='gray')

def is_black(img):
	return np.all(img == 0)

def is_white(img):
	return np.all(img == 255)

def getNewImgMultipliedBy(img, scale_factor):
	# TODO img * 2 does not work for some reason. Maybe investigate why
	return np.multiply(img, np.full(img.shape, scale_factor))

def test_all_combinations(imgs, img_names, pdf_mode=True, output_folder=""):
	for i in range(len(imgs)):
		# imgA = imgs[i]
		imgA = getNewImgMultipliedBy(img, 1)
		imgA_name = img_names[i]
		# save_img(imgA, imgA_name, "255_branch")
		# show_img(imgA)
		# assert is_from_0_to_255(imgA), "IMGA PROBLEM"
		# print(np.min(imgA))
		# print(np.max(imgA))
		for j in range(len(imgs)):
			try:
				imgB = imgs[j]
				imgB_name = img_names[j]
				# assert is_from_0_to_255(imgB), "IMGB PROBLEM"
				# print(type(imgs[j]))
				# print(np.max(imgs[j]))
				# print(type(imgB))
				# print(np.max(imgB))
				# show_img(imgA)
				# show_img(imgB)
				if pdf_mode==True:
					matched_imgA = GHM(imgA, imgB)
				else:
					# ROCKYFIX no more cdfGHM. Also get booleans right: either pdfTrue or cdfTrue
					matched_imgA = cdfGHM(imgA, imgB)
				save_img(matched_imgA, imgA_name + "_" + imgB_name, "255_branch")
				# show_img(matched_imgA)
			except Exception as err:
				filepath = "255_branch/" + imgA_name + "_" + imgB_name + "ERROR"
				print(err)
				# print(traceback.format_exc())
				f = open(filepath + ".txt", "w")
				f.write(str(err))
				# f.write(traceback.format_exc())
				f.close()

def load_imgs():
	black = read_and_check_img("black.jpg")
	# black = create_black()
	# assert is_black(black), "black image is not actually black (i.e. not all pixel values are 0)"

	grayish = read_and_check_img("grayish.jpg")

	house = read_and_check_img("house.jpg")

	streetlights = read_and_check_img("streetlights.jpg")

	white = read_and_check_img("white.jpg")
	# assert is_white(white), "white image is not actually white (i.e. not all pixel values are 255)"

	imgs = [black, grayish, house, streetlights, white]
	img_names = ['black', 'grayish', 'house', 'streetlights', 'white']
	# for i in range(len(imgs)):
	# 	assert is_from_0_to_255(imgs[i]), img_names[i] + " not in 0-255 range"
	return imgs, img_names

imgs, img_names = load_imgs()
for i in range(len(imgs)):
	img = imgs[i]
	img_name = img_names[i]

	print("$$$$")

	print("Original image")
	print(img)
	show_img(img)

	print("Image Doubled")
	imgDoubled = getNewImgMultipliedBy(img, 2)
	print(imgDoubled)
	show_img(imgDoubled)

	save_img(img, img_name, "255_branch")


test_all_combinations(imgs, img_names)


# a = np.array([[1]])
# b = np.array([[1,2],[3,4]])

# print(type(a))
# print(type(b))
# print(a)
# print(b)
# print(a*2)
# print(b*2)
# print(np.multiply(a, np.full(a.shape, 2)))
# print(np.multiply(b, np.full(b.shape, 2)))















