from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from PIL import Image
import cv2

import datetime
import time
import argparse
import pyflow




class videoProcessor:

	def __init__(self,
					video_path = "action_annotation/P_01.mp4",
					action_list = "action_annotation/P_01.mp4"
					)

 #code here
def Main():

	successFlag = True

	#The shape we want our array to be in
	RGB_Input = np.zeros((1,64,224,224,3))
	FLOW_Input = np.zeros((1,64,224,224,2))

	vidCap = cv2.VideoCapture("Razan washing dishes.mp4")

	# Get to correct starting frame
	skipTillSegment(21,vidCap,30)

	frameNum = 0

	#Have to read 1 frame in ahead of time to calculate optical flow
	successFlag, frameOld = vidCap.read()
	frameOld = cv2.cvtColor(frameOld, cv2.COLOR_BGR2RGB)
	im = Image.fromarray(frameOld)
	frameOld = processImage(frameNum,im)

	while(successFlag and frameNum < 64):
		#Capture frame by frame (frame is a numpy array)
		successFlag, frame = vidCap.read()
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		#Shows that is a numpy.ndarray
		print(type(frame))
		print(frame.shape)
		print(successFlag)
		im = Image.fromarray(frame)
		frame = processImage(frameNum,im)
		RGB_Input[0][frameNum] = frame


		flowImage = calcOpticalFlow(frame,frameOld,frameNum) #flow of image
		FLOW_Input[0][frameNum] = flowImage

		frameOld = frame # for calculating optical flow, you need previous frame
		frameNum += 1


	
	#needs to be an unsigned int for the bytes
	#im = Image.fromarray((RGB_Input[0][149]).astype('uint8'))
	#im.show()

	#Generate the frames and save them
	makeTrainingInput(RGB_Input,FLOW_Input)

	print("Before Scaling: " + str(time.time()))
	#Scale between -1 and 1
	RGB_Input *= (2.0/255)
	RGB_Input -= 1
	print("After: " + str(time.time()))

	#Save the RGB_Input and Optical Flow input
	np.save('examples/outFlow.npy', FLOW_Input)
	np.save('examples/outRGB.npy', RGB_Input)







#Processes the current image and stores it into the RGB_Input numpy array
#
#RGB_Input - the numpy array (input to the neural network 1,79,224,224,3
#frameNum - the frame number that we are processing, somewhere between 1-79
#im - the Image object from Pillow library corresponding to the current frame
def processImage(frameNum, im):


	#Make smallest image dimension 256
	if(im.width > im.height):
		im.thumbnail((im.width,256), Image.BILINEAR);
	else:
		im.thumbnail((256,im.height), Image.BILINEAR);

	print(im.size)




	#Crop a 224x224 image out of center and convert to npy array
	horizOffset = (im.width - 224) / 2
	vertOffset = (im.height - 224) / 2

	im = im.crop((horizOffset,vertOffset,
				im.width-horizOffset, im.height-vertOffset))
	#im.show()

	print(im.size)

	imageData = np.array(im)
	#print(imageData.shape)

	#read in the image data (numpy array) into the correct frame index
	return imageData
	#NetworkInput[0][frameNum] = imageData;
	#print(testArray[0][0][0])

	# Some code to help vizualize what is happening with higher
	# dimensionality arrays,TLDR: it's just groups of groups of
	# groups of 2D arrays
	# thinkingDimensions = np.zeros((2,2,2,2,2))
	# print(thinkingDimensions)



#Calculates the optical flow
def calcOpticalFlow(im1, im2, count):
	im1 = im1.astype(float) / 255.
	im2 = im2.astype(float) / 255.

	# Flow Options:
	alpha = 0.012
	ratio = 0.75
	minWidth = 20
	nOuterFPIterations = 7
	nInnerFPIterations = 1
	nSORIterations = 30
	colType = 0  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))

	s = time.time()
	u, v, im2W = pyflow.coarse2fine_flow(
	    im1, im2, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,
	    nSORIterations, colType)
	e = time.time()
	print('Time Taken: %.2f seconds for image of size (%d, %d, %d)' % (
	    e - s, im1.shape[0], im1.shape[1], im1.shape[2]))
	flow = np.concatenate((u[..., None], v[..., None]), axis=2)
	return flow
	#np.save('examples/outFlow.npy'+str(count), flow)



#Skips until segmented part of video (in seconds)
#startingTime - when in the video the segment starts (pos. integer)
#vidCap - the inputstream of video from a file.
#fps - the fps of the video, to calculate how many frames to skip
def skipTillSegment(startingTime, vidCap, fps):

	currentFrame = 0

	startFrame = fps * startingTime

	successFlag = True

	#Stop 1 frame before startframe so that we have the initial frame to calculate optical flow with
	while(successFlag and currentFrame < startFrame-1):
		#Skip frame by frame
		successFlag, frame = vidCap.read()
		currentFrame += 1



#rgb - [1][64][224][224][3] numpy array
#flow- [1][64][224][224][2] numpy array
#This method is to turn these numpy arrays into images. 64 rgb frames, 64 flowx frames, 64 flowy frames
#Then save these frames into a directory.
def makeTrainingInput(rgb,flow):

	#loops through the frames in the form of a numpy array and makes imgs
	for i in range(len(rgb[0])):	
		temp = flow[0][i] # now it is [224][224][2]
		flowx = temp[:,:,0] #Get the flowx portion
		flowy = temp[:,:,1] 

		imFlowx = Image.fromarray(flowx.astype('uint8'),'P')
		imFlowy = Image.fromarray(flowy.astype('uint8'),'P')
		im = Image.fromarray((rgb[0][i]).astype('uint8'))

		#Save the three images
		imFlowx.save("inputData/flow_x" + str(i) + ".png")
		imFlowy.save("inputData/flow_y" + str(i) + ".png")
		im.save("inputData/img_" + str(i) + ".png")




#Showing
if __name__ == "__main__":
	Main()