import cv2 
import numpy as np
import sys
import time

def blur(box, img):
	x = box[0]
	w = box[2] - box[0]
	y = box[1]
	h = box[3] - box[1]
	roi = img[y:y+h, x:x+w]
	blur = cv2.blur(roi,(511,511))
	img[y:y+h, x:x+w] = blur
	return img
	