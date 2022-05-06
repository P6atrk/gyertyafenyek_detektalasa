import cv2
import numpy as np

def find_candles(threshold_val, fill_diff_val, img):
	if ADD_SALT_PEPPER_NOISE:
		img = add_salt_pepper_noise(img)
	if ADD_ADDITIVE_NOISE:
		img = add_additive_noise(img)
	img_original = img.copy()
	
	#img = noise_reduction(img) # ez nem tul hasznos, csak ront a kereses minosegen
	#brightest = threshold_brightest_spot(threshold_val, img)
	brightest = threshold_brightest_spot(threshold_val, img)
	mask = flood_fill(brightest, fill_diff_val, img)
	mask = morphology_filter(mask)

	img_original, contours = find_contours(mask, img_original, SHOW_CONTOURS)
	count = count_candles(mask)
	img_original = circle_candles(contours, img_original)
	print("Gyertyak szama: ", count)
	if SHOW_MASK:
		cv2.imshow("mask", mask)
	cv2.imshow("img", img_original)

def threshold_trackbar(val):
	global img, threshold_val, fill_diff_val
	threshold_val = val
	img_copy = img.copy()
	find_candles(threshold_val, fill_diff_val, img_copy)

def flood_fill_trackbar(val):
	global img, threshold_val, fill_diff_val
	fill_diff_val = val
	img_copy = img.copy()
	find_candles(threshold_val, fill_diff_val, img_copy)

def threshold_brightest_spot(val, img):
	img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
	y, u, v = cv2.split(img_yuv)
	max_val = np.max(y)
	_, y_thresh = cv2.threshold(y, max_val - val, max_val, cv2.THRESH_TOZERO)
	if SHOW_BRIGHTEST_SPOTS:
		cv2.imshow("brightest_spots", y_thresh)
	return y_thresh

def flood_fill(src, fill_diff, img):
	mask = np.zeros((src.shape[0] + 2, src.shape[1] + 2), dtype=np.uint8)
	non_zeros = np.transpose(np.nonzero(src))[:, ::-1]
	lo_diff = [fill_diff, fill_diff, fill_diff]
	hi_diff = [fill_diff, fill_diff, fill_diff]
	for i in range(len(non_zeros)):
		_, src, mask, _ = cv2.floodFill(img, mask, non_zeros[i], None, lo_diff, hi_diff, 8 | (255 << 8) | cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE)
	mask = mask[1:len(mask) - 1, 1:len(mask[0]) - 1] # levágja a külső részét a masknak amik 1esek
	return mask

def find_contours(src, img, draw_contours):
	contours, hierarchy = cv2.findContours(src, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	if draw_contours:
		cv2.drawContours(img, contours, -1, CONTOUR_COLOR, 2)
	return img, contours

def count_candles(src):
	retval, _ = cv2.connectedComponents(src)
	return retval - 1

def morphology_filter(src):
	struct = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
	mask = cv2.dilate(cv2.erode(cv2.dilate(src, struct), struct, iterations=4), struct, iterations=1)
	return mask

def circle_candles(contours, img):
	if contours is None:
		return img
	for i in range(len(contours)):
		(x, y), radius = cv2.minEnclosingCircle(contours[i])
		center = (int(x),int(y))
		radius = int(radius)
		img = cv2.circle(img, center, radius, CIRCLE_COLOR, 2)
	return img

def add_salt_pepper_noise(img):
	n = int(img.shape[0] * img.shape[1] * 0.01)

	for _ in range(1, n):
		i = np.random.randint(0, img.shape[1])
		j = np.random.randint(0, img.shape[0])
		value = np.random.randint(0, 255)

		if img.ndim == 3:
			img[j, i] = [value, value, value]

	return img


def add_additive_noise(img):
	img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
	y, u, v = cv2.split(img_yuv)

	noise = np.zeros(img.shape[:2], np.uint16)
	cv2.randn(noise, -60, 60)

	y_noise = cv2.add(y, noise, dtype=cv2.CV_8UC1)
	
	img = cv2.merge([y_noise, u, v])
	img = cv2.cvtColor(img, cv2.COLOR_YUV2BGR)
	return img

def noise_reduction(img):
	img = cv2.blur(img, (5, 5))
	return img

# kepek betoltese
img0 = cv2.imread("573px-Candles_in_the_church.jpg")
#img1 = cv2.imread("640px-Basilique_Notre-Dame_Fourviere_crypte_reconnaissance_a_marie.jpg")
#img2 = cv2.imread("640px-Candles_in_the_dark.jpg")
#img3 = cv2.imread("gyertyak.jpg")
#img4 = cv2.imread("get-the-look-the-rustic-chic-table-c.jpg")
#img5 = cv2.imread("LibertePot_collection_2020_12x8_070720_1200x.jpg")
#img6 = cv2.imread("aak-tefacid-sustainable-stearic-acid-16x9-1500-matrix.jpg")
#img7 = cv2.imread("aroma-candles-500x500.jpg")
#img8 = cv2.imread("510557_shutterstock_540141442.jpg")

SHOW_CONTOURS = False # Konturok kirajzolasa az eredeti kepen
ADD_SALT_PEPPER_NOISE = False # Noise hozzadasa a vizsgalando kephez
ADD_ADDITIVE_NOISE = False # Szinten
SHOW_MASK = False # A talalt fenyek maszkja
SHOW_BRIGHTEST_SPOTS = False # A legfenyesebb pontok kirajzolasa, ebbol lesz a maszk

SHOW_TRACKBARS = False # Lehet a Threshold-ot es a Floodfill fuggveny bemeneti erteket allitani

CONTOUR_COLOR = [255, 0, 255]
CIRCLE_COLOR = [0, 255, 0]

threshold_val = 0
fill_diff_val = 0
img = None

if SHOW_TRACKBARS:
	img = img0 # img0 modositasaval mas kepet lehet trackbar-okkal megjeleniteni
	find_candles(threshold_val, fill_diff_val, img)
	cv2.createTrackbar("threshold", "img", 0, 200, threshold_trackbar)
	cv2.createTrackbar("value_diff", "img", 0, 50, flood_fill_trackbar)
else: # valtoztatni lehet melyik kep jelenik meg, eloszor be KELL oket tolteni fentebb
	find_candles(4, 14, img0)
	#find_candles(14, 1, img1)
	#find_candles(2, 1, img2)
	#find_candles(3, 1, img3)
	#find_candles(4, 1, img4)
	#find_candles(1, 8, img5)
	#find_candles(19, 5, img6)
	#find_candles(10, 2, img7)
	#find_candles(16, 15, img8)

cv2.waitKey(0)

cv2.destroyAllWindows()