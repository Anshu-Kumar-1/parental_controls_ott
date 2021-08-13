from nudenet import NudeDetector
import matplotlib.pyplot as plt
from blur_image import *
from config import unsafe
import os
import sys

def read_image(img):
		w,h, _ = img.shape
		if(w>320 or h>255):
			img = cv2.resize(img, (320, 255),interpolation = cv2.INTER_NEAREST)
		out = detection(img)
		return out

def detection(img):
	detector = NudeDetector()
	out = detector.detect(img)
	return out

def main(img):
	if os.path.exists(img):
		img = cv2.imread(sys.argv[1])
	out = read_image(img)
	blurred = img
	print(out)
	cv2.imshow("Original Image",img)
	for i in range(len(out)):
		for k in out[i].keys():
			if(k == 'label' and out[i][k] in unsafe):
				# print(i)
				box = out[i]['box']
				blurred = blur(box,img)
	cv2.imshow("Blurred", blurred)
	cv2.waitKey(2000)
	cv2.destroyAllWindows()
 

if __name__ == '__main__':
	img = sys.argv[1]
	main(img)


