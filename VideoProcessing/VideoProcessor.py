from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from PIL import Image
import cv2
import collections
import datetime
import time
import argparse
#import pyflow



#Base class for processing videos. This video processor is made for processing the ADL dataset
#but can be reused for other purposes.
class VideoProcessor:

	def __init__(self,
					video_path = "./ADL_Dataset/P_01.mp4",
					action_list_path = "action_annotation/overlap_labels_final.txt",
					action_annotation_path = "action_annotation/P_01.txt",
					num_frames = 64,):

		self.video_path = video_path
		self.action_list_path = action_list_path
		self.action_annotation_path = action_annotation_path
		self.num_frames = 64
		self.action_list = {} # Will populate with the action label




		#Array index corresponds to label index, string corresponds name of label
		with open(action_list_path) as fin:
			for line in fin.readlines():
				label = line.split("(")[1].split(")")[0] # Get the label
				self.action_list[int(label)] = line.split("'")[1]

		# for debugging
		# for key, value in self.action_list.items():
		# 	print(str(key) + " " + str(value))

		self.annotations = self.get_annotations()





	#Creates Directory with all the videos processed as RGB + Flowx + Flowy at the frame rate specified
	#in the constructor
	def process_for_I3D_input():
		
		return 0


	#Returns named tuple of the annotations file
	#start_time - int, end_time - int, label - String
	#time is converted from mm:ss into seconds.
	#label is converted from index to the correpsonding string label name
	def get_annotations(self):

		Annotation = collections.namedtuple("annotation", "start_time end_time label")

		annotations = []


		#Process the annotation file.
		with open(self.action_annotation_path) as fin:
			for line in fin.readlines():
				temp = line.split(" ")

				start_time = 0
				end_time = 0
				label = int(temp[2])

				start_time += (int(temp[0][0])*10 + int(temp[0][1]))*60
				start_time += (int(temp[0][3])*10 + int(temp[0][4]))

				end_time += (int(temp[1][0])*10 + int(temp[1][1]))*60
				end_time += (int(temp[1][3])*10 + int(temp[1][4]))

				if(label in self.action_list):
					label = self.action_list[label] #Account for index starting at 0
					annotations.append(Annotation(start_time, end_time, label))

		#For debugging
		# for annotation in annotations:
		# 	print(str(annotation.start_time) + " " + str(annotation.end_time) + " " + annotation.label)

		return self.get_annotations




	def process_video_for_training(self):


		successFlag = True

		#The shape we want our array to be in
		RGB_Input = np.zeros((1,64,224,224,3))
		FLOW_Input = np.zeros((1,64,224,224,2))

		vidCap = cv2.VideoCapture(self.video_path)

		# Get to correct starting frame
		self.skipTillSegment(21,vidCap,30)

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

	def process_video_for_evaluation():
		return 0




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
	def skipTillSegment(self,startingTime, vidCap, fps):

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