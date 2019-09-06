import cv2
import numpy as np
import imutils
import os
import time
from recognition.num.find_number import my_recognizer as num_recognizer
from recognition.alpha.find_number import my_recognizer as char_recognizer
os.system('color 0a')

def main():
	try:
		print('>> Loading Code')
		begin()
		input('Press Enter to continue...')
	except Exception as e:
		print(e)
		input('Press Enter to continue...')

def imshow(img):
	cv2.imshow('test', img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def my_thresholding(image):
	#adding white padding to each individual cropped character
	shape=image.shape
	w=shape[1]
	h=shape[0]
	base_size=h+20,w+20,3
	#make a 3 channel image for base which is slightly larger than target img
	base=np.zeros(base_size,dtype=np.uint8)
	cv2.rectangle(base,(0,0),(w+20,h+20),(255,255,255),30)#really thick white rectangle
	base[10:h+10,10:w+10]=image
	#cv2.imshow('bordered', base)
	#cv2.waitKey(0)
	#====================================================================================
	return base

def my_num_ocr(image):
	#from recognition.num.find_number import my_recognizer as num_recognizer
	try:
		number = num_recognizer.find(image)
		#print(str(number))
		#os.system('pause')
		return number
	except Exception as e:
		print(e)

def my_alpha_ocr(image):
	#from recognition.num.find_number import my_recognizer as char_recognizer
	try:
		alpha = char_recognizer.find(image)
		#print(str(number))
		#os.system('pause')
		return alpha
	except Exception as e:
		print(e)

def begin():
	print('>> Reading image')
	img = cv2.imread('raw/mstl2.png')

	print('>> Resizing input image by width = 900px')
	img = imutils.resize(img, width=900)
	cv2.imwrite('temp_photos/1.input_resized.png', img)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray, (11, 11), 2)
	#imshow(blurred)
	kernel = np.ones((25,25), np.uint8)
	img_dilation = cv2.dilate(blurred, kernel, iterations=1)

	gray = img_dilation
	#imshow(gray)
	# initialize a rectangular (wider than it is tall) and square
	# structuring kernel
	rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 1))
	
	# apply a tophat (whitehat) morphological operator to find light
	# regions against a dark background (i.e., the credit card numbers)
	tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)
	#imshow(tophat)
	# compute the Scharr gradient of the tophat image, then scale
	# the rest back into the range [0, 255]
	gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0,
		ksize=-1)
	gradX = np.absolute(gradX)
	(minVal, maxVal) = (np.min(gradX), np.max(gradX))
	gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
	gradX = gradX.astype("uint8")
	print('>> Finding vertical scharr gradient')
	cv2.imwrite('temp_photos/2.input_vertical_scharr_gradient.png', gradX)
	#imshow(gradX)

	# apply a closing operation using the rectangular kernel to help
	# cloes gaps in between credit card number digits, then apply
	# Otsu's thresholding method to binarize the image
	gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)

	thresh = cv2.threshold(gradX, 0, 255,
		cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

	sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 20))
	# apply a second closing operation to the binary image, again
	# to help close gaps between credit card number regions
	thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)
	print('>> Generating Horizontal projection to find text')
	cv2.imwrite('temp_photos/3.input_Horizontal_projection.png', thresh)
	#imshow(thresh)

	# find contours in the thresholded image, then initialize the
	# list of digit locations
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)

	#finding the largest area rectangular contour
	areas = [cv2.contourArea(c) for c in cnts]
	max_index = np.argmax(areas)
	cnt=cnts[max_index]

	x,y,w,h = cv2.boundingRect(cnt)
	cv2.rectangle(img,(x-20,y-20),(x+w+20,y+h+20),(0,255,0),2)
	#imshow(img)

	#cropped_img = img[y-19:y+h+20, x-19:x+w+20]
	cropped_img = img[y-10:y+h+10, x-10:x+w+10]
	#cropped_img = img[y:y+h, x:x+w]
	cropped_img = imutils.resize(cropped_img, height=100)
	#imshow(img)
	print('>> Cropping Text Roi')
	cv2.imwrite('temp_photos/4.cropped.png', cropped_img)
	#cv2.imshow('cropped',cropped_img)

	#================= Character segmentation ================================
	image = cropped_img.copy()
	dst = cv2.fastNlMeansDenoisingColored(image.copy(), None, 10, 10, 7, 15)
	print('>> Removing Noise from cropped Text Roi')
	cv2.imwrite('temp_photos/5.cropped_denoise.png', dst)
	#cv2.imshow('denoise', dst)
	#grayscale 
	gray = cv2.cvtColor(dst,cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray, (3, 7), 13)
	#cv2.imshow('blur', blurred)

	edge = cv2.Canny(blurred,80,30)
	print('>> Finding canny edge for text segmentation')
	cv2.imwrite('temp_photos/6.cropped_canny_edge.png', edge)
	#cv2.imshow('canny', edge)
	edge = np.uint8(edge)

	kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(10,500))
	closing = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, kernel)
	print('>> Generating vertical projection')
	cv2.imwrite('temp_photos/7.cropped_vertical_projection.png', closing)
	#cv2.imshow('vertical projection', closing)
	#cv2.waitKey(0)

	kernel = np.ones((11,9), np.uint8)
	img_dilation = cv2.dilate(edge, kernel, iterations=1)
	#cv2.imshow('dilate',img_dilation)
	#cv2.imwrite('dilate.png', img_dilation)

	edge = img_dilation
	#find contours 
	ctrs,_ = cv2.findContours(closing.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
	#sort contours 
	sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
	idx = 1
	segments = {}
	for i, ctr in enumerate(sorted_ctrs): 
	    # Get bounding box 
	    x, y, w, h = cv2.boundingRect(ctr) 
	    
	    # Getting ROI 
	    roi = image[y:y+h, x:x+w] 
	    # show ROI 
	    #cv2.imshow('segment no:'+str(i),roi) 
	    #cv2.rectangle(image,(x,y),( x + w, y + h ),(0,255,0),2) 
	    #cv2.waitKey(0) 
	    if h>20 and w>10:
	    	cv2.rectangle(image,(x-7,y+10),( x + w, y + h-10 ),(0,255,),2)
	    	segments.update({idx:(x-7,y+10,x+w,y+h-10)})
	    	idx+=1
	print('>> Segmenting individual Character')
	cv2.imwrite('temp_photos/8.segmented_text.png', image)
	cv2.imshow('segmented text',image)
	#cv2.waitKey(0)

	#==============Now we will crop out each character segment for recognition ========
	#print(segments)
	text_segment = {}
	for i in range(1,len(segments)+1):
		x,y,w,h = segments[i]
		cropped_segment = cropped_img[y:h, x:w]
		#======================= Now applying thresholding on each character ==========
		cropped_segment = my_thresholding(cropped_segment)
		#==============================================================================
		text_segment.update({i:cropped_segment})
		cv2.imwrite('temp_photos/segments/'+str(i)+'.png',cropped_segment)

	#imshow(cropped_segment)
	idx = 1
	final_ocr = ''
	for i in range(1,len(segments)+1):
		if idx == 1 or idx == 4 or idx == 5:#false
			#block for alphabet recognition
			
			any_char = text_segment[i] # filtering alpha characters as per position
			#recognizing a alphabet
			alpha_char = my_alpha_ocr(any_char)
			#print('position: '+str(i)+', Recognized: '+str(alpha_char))
			final_ocr = str(final_ocr+str(alpha_char))
			
			#print('position: '+str(i)+', Recognized: X')
			#final_ocr = str(final_ocr+'X') #filing with x when not recognizing
			#cv2.imshow('any char', any_char)
			#cv2.waitKey(0)
			idx+=1
		else:
			#block for number recognition

			any_char = text_segment[i] # filtering number characters as per position
			#recognizing a number
			num_char = my_num_ocr(any_char)
			#print('position: '+str(i)+', Recognized: '+str(num_char))
			final_ocr = str(final_ocr+str(num_char))

			#cv2.imshow('any char', any_char)
			#cv2.waitKey(0)
			idx+=1
	print()
	print('OCR: '+final_ocr)
	cv2.waitKey(0)

#==============================
if __name__ == '__main__':
	main()