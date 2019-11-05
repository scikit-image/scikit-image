from GHM import GHM
from GHMHelperFuncs import read_and_check_img

house = read_and_check_img("house.jpg")
streetlights = read_and_check_img("streetlights.jpg")

matched = GHM(house, streetlights)