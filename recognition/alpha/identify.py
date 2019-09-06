import os
import cv2

from find_number import my_recognizer as char_recognizer
try:
	image = cv2.imread('1.png')
	number = char_recognizer.find(image)
	print(str(number))
	os.system('pause')
except Exception as e:
	print(e)
	os.system('pause')