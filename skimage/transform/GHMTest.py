import numpy as np

from GHM import GHM
from GHMHelperFuncs import read_and_check_img

black = read_and_check_img("black.jpg")
grayish = read_and_check_img("grayish.jpg")
house = read_and_check_img("house.jpg")
streetlights = read_and_check_img("streetlights.jpg")
white = read_and_check_img("white.jpg")
imgs = [black, grayish, house, streetlights, white]
img_names = ['black', 'grayish', 'house', 'streetlights', 'white']

def is_black(img):
	return np.all(img == 0)
def is_white(img):
	return np.all(img == 255)

def test_all_combinations_pdf(imgs, img_names):
	for i in range(len(imgs)):
		for j in range(len(imgs)):
			try:
				imgA = imgs[i]
				imgA_name = img_names[i]
				imgB = imgs[j]
				imgB_name = img_names[j]
				matched_imgA = GHM(imgA, imgB)
				filepath = "255_branch/" + imgA_name + "_" + imgB_name
				np.save(filepath, matched_imgA)
				plt.imsave(filepath + ".jpg", matched_imgA, cmap='gray')
			except Exception as err:
				filepath = "255_branch/" + imgA_name + "_" + imgB_name + "ERROR"
				print(err)
				# print(traceback.format_exc())
				f = open(filepath + ".txt", "w")
				f.write(str(err))
				f.write(traceback.format_exc())
				f.close()


# matched = GHM(house, streetlights)
# test_all_combinations_pdf(imgs, img_names)
print("****")
print(is_black(black))
print(black.shape)
print("****")
print(is_white(white))
print(white.shape)