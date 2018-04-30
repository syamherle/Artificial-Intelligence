#!/usr/bin/python
#
# Mountain ridge finder
# Based on skeleton code by D. Crandall, Oct 2016
#
# Part 1: For the part 1, the idea behind formualting this problem was to start from column 0 and find the max gradient at that column, then suppose I found
# the max gradient at row number 5 for column 0, then moving on to column 1, I'll assign the max probability of 1 to the row number 5 of column 1 because we are
# assuming that the ridge is a straight line and as we are moving away from row number 5, i.e we are considering the vertical distance from row number 5, as the
# distance increases, our probability decreases. Now, we found the maximum probaility of a pixel at say row number 10 of column 1, now keeping this row number 10
# in mind, we will calculate the next columns max probability. In this sense, we will calculate the max probability of each pixel at each column and then plot it.
#
# Part 2: This part uses the same approach as that of the part 1 with some modifications. I translated the problem into a Hidden Markov Model.
# What the algorith does is that it finds a random column and then try to find the max gradient in that column and then use the max probability of the current
# column to find the maximum probability of the next column. The initial probability is the max edge strength of a random column chosen, the transitional
# probability will be the (1 / distance where distance will be the vertical distance of the row number as discussed in part 1, the emission probability
# will be the edge strength of a particular point (x,y) where x and y are row and column number respectively.
# Now this is just one sample, we are incorporating the Gibbs sampling method, where we are generating 100 samples for random columns and then try to estimate
# the ridge line of the mountain. After we have a dataset of 100 ridge lines, we plot the pixel with maximum frequency per column i.e the pixel which comes
# most in each column in all the samples is plotted on the image.
#
# Part 3: In this part we are given a human input where the a point (x, y) is assumed to lie on the ridge line and then we try to map the ridge line. My approach
# for this part uses the same algorithm as in part 2 but without the sampling part and the part where I choose a random column number, in this case we use the 
# point provided by the human as our initial probability and then try to use the same method as in part 2 to calculate the max probability in a column and then
# map the ridge line on the image.
#
# One assumption that I have made in part3 is that I am assuming that the the ridge line is not fluctuating that much, so if for a column number 5, we find the max
# probability at row number 10, then for column number 6, I am assuming the max probability can be 2 rows above or 2 rows below. Beyond this range, I am scaling down
# the emission probability of the rest columns because otherwise it will again try to map the points which has a higher edge strength than the one we need.

from PIL import Image
from numpy import *
from scipy.ndimage import filters
from scipy.misc import imsave
from random import randint
import sys


class mountainRidgeFinding:

	def __init__(self, edge_strength, gt_row, gt_col):
		self.edge_strength = edge_strength
		self.gt_row = gt_row
		self.gt_col = gt_col

	def calculateRandomGradient(self):
		firstRandom = randint(0, edge_strength.shape[1]-1)
		
		maxGradient = []
		sample = []
		
		for i in range(0 , len(edge_strength)):
			maxGradient.append(edge_strength[i][firstRandom])
		sample.append(maxGradient.index(max(maxGradient)))
		return(firstRandom, sample[0])

	def sample2(self):
		sampleList = []
		xCoord = edge_strength.shape[1]
		yCoord = edge_strength.shape[0]
		for i in range(0, xCoord):
			newList = []
			for j in range(0, yCoord):
				if(i == 0):
					newList.append(edge_strength[j][i])
				else:
					lastRow = sampleList[len(sampleList)-1]
					if(lastRow > j):
						distanceFromLastRow = lastRow - j
						newList.append(edge_strength[j][i] * (1.0 / distanceFromLastRow ))
					elif(lastRow == j):
						newList.append(edge_strength[j][i])
					else:
						distanceFromLastRow = j - lastRow
						newList.append(edge_strength[j][i] * (1.0 / distanceFromLastRow))
			sampleList.append(newList.index(max(newList)))
		return sampleList

	def sample3(self):
		sampleList = []
		yCoord = edge_strength.shape[0]
		finalXCoord = edge_strength.shape[1]
		coordTuple = self.calculateRandomGradient()
		xCoord = coordTuple[0]
		sampleList.insert(0, coordTuple[1])
		for i in range(xCoord-1 , -1, -1):
			newList = []
			gradientSum = 0
			for k in range(0, yCoord):
				gradientSum += edge_strength[k][i]
			for j in range(0, yCoord):
				lastRow = sampleList[0]
				if(lastRow > j):
					distanceFromLastRow = lastRow - j
					newList.append((edge_strength[j][i] / gradientSum) * (1.0 / distanceFromLastRow))
				elif(lastRow == j):
					newList.append(edge_strength[j][i] / gradientSum)
				else:
					distanceFromLastRow = j - lastRow
					newList.append((edge_strength[j][i] / gradientSum) * (1.0 / distanceFromLastRow))
			sampleList.insert(0, newList.index(max(newList)))
		
		for i in range(xCoord+1, finalXCoord):
			newList = []
			gradientSum = 0
			for k in range(0, yCoord):
				gradientSum += edge_strength[k][i]
			for j in range(0, yCoord):
				lastRow = sampleList[len(sampleList)-1]
				if(lastRow > j):
					distanceFromLastRow = lastRow - j
					newList.append((edge_strength[j][i] / gradientSum) * (1.0 / distanceFromLastRow))
				elif(lastRow == j):
					newList.append(edge_strength[j][i] / gradientSum) 
				else:
					distanceFromLastRow = j - lastRow
					newList.append((edge_strength[j][i] / gradientSum) * (1.0 / distanceFromLastRow))
			sampleList.append(newList.index(max(newList)))
		return sampleList

	def sample4(self):
		sampleList = []
		yCoord = int(gt_row)
		xCoord = int(gt_col)
		finalXCoord = edge_strength.shape[1]
		finalYCoord = edge_strength.shape[0]
		sampleList.insert(0, yCoord)
		for i in range(xCoord-1, -1, -1):
			newList = []
			gradientSum = 0
			for k in range(0, finalYCoord):
				gradientSum += edge_strength[k][i]
			for j in range(0, finalYCoord):
				lastRow = sampleList[0]
				if(lastRow > j):
					distanceFromLastRow = lastRow - j
					if(distanceFromLastRow > 2):
						newList.append(((edge_strength[j][i] / gradientSum) * 0.01) * (1.0 / distanceFromLastRow))
					else:
						newList.append((edge_strength[j][i] / gradientSum) * (1.0 / distanceFromLastRow))
				elif(lastRow == j):
					newList.append(edge_strength[j][i] / gradientSum)
				else:
					distanceFromLastRow = j - lastRow
					if(distanceFromLastRow > 2):
						newList.append(((edge_strength[j][i] / gradientSum) * 0.01) * (1.0 / distanceFromLastRow))
					else:
						newList.append((edge_strength[j][i] / gradientSum) * (1.0 / distanceFromLastRow))
			sampleList.insert(0, newList.index(max(newList)))
		for i in range(xCoord+1, finalXCoord):
			newList = []
			gradientSum = 0
			for k in range(0, finalYCoord):
				gradientSum += edge_strength[k][i]
			for j in range(0, finalYCoord):
				lastRow = sampleList[len(sampleList) - 1]
				if(lastRow > j):
					distanceFromLastRow = lastRow - j
					if(distanceFromLastRow > 2):
						newList.append(((edge_strength[j][i] / gradientSum) * 0.01) * (1.0 / distanceFromLastRow))
					else:
						newList.append((edge_strength[j][i] / gradientSum) * (1.0 / distanceFromLastRow))
				elif(lastRow == j):
					newList.append(edge_strength[j][i] / gradientSum)
				else:
					distanceFromLastRow = j - lastRow
					if(distanceFromLastRow > 2):
						newList.append(((edge_strength[j][i] / gradientSum) * 0.01) * (1.0 / distanceFromLastRow))
					else:
						newList.append((edge_strength[j][i] / gradientSum) * (1.0 / distanceFromLastRow))
			sampleList.append(newList.index(max(newList)))
		return sampleList
					
				 
		

	def mainClass(self):
		listOfSamples = []
		for i in range(0, 100):
			listOfSamples.append(self.sample3())
		samples2 = self.sample2()
		humanSample = self.sample4()
		return (listOfSamples, samples2, humanSample)

# calculate "Edge strength map" of an image
#
def edge_strength(input_image):
    grayscale = array(input_image.convert('L'))
    filtered_y = zeros(grayscale.shape)
    filters.sobel(grayscale,0,filtered_y)
    return filtered_y**2

# draw a "line" on an image (actually just plot the given y-coordinates
#  for each x-coordinate)
# - image is the image to draw on
# - y_coordinates is a list, containing the y-coordinates and length equal to the x dimension size
#   of the image
# - color is a (red, green, blue) color triple (e.g. (255, 0, 0) would be pure red
# - thickness is thickness of line in pixels
#
def draw_edge(image, y_coordinates, color, thickness):
    for (x, y) in enumerate(y_coordinates):
        for t in range( max(y-thickness/2, 0), min(y+thickness/2, image.size[1]-1 ) ):
            image.putpixel((x, t), color)
    return image

# main program
#
(input_filename, output_filename, gt_row, gt_col) = sys.argv[1:]

# load in image 
input_image = Image.open(input_filename)

# compute edge strength mask
edge_strength = edge_strength(input_image)
imsave('edges.jpg', edge_strength)

# You'll need to add code here to figure out the results! For now,
# just create a horizontal centered line.
#ridge = [ edge_strength.shape[0]/2 ] * edge_strength.shape[1]

ridge = []

for i in range(0, len(edge_strength[0])):
	newList = []
	for j in range(0, len(edge_strength)):
		newList.append(edge_strength[j][i])
	ridge.append(newList.index(max(newList)))


mountain = mountainRidgeFinding(edge_strength, gt_row, gt_col)
sampleList = mountain.mainClass()

sampleListOld = sampleList[0]
sampleListNew = sampleList[1]
humanSample = sampleList[2]

newRidge2 = []
for i in range(0, len(sampleListOld[0])):
	newDict = {}
	for j in range(0, len(sampleListOld)):
		if(newDict.get(sampleListOld[j][i]) != None):
			value = newDict[sampleListOld[j][i]]
			newDict[sampleListOld[j][i]] = value + 1
		else:
			newDict[sampleListOld[j][i]] = 1
	newRidge2.append(max(newDict, key=(lambda key: newDict[key])))

# output answer
imsave(output_filename, draw_edge(input_image, ridge, (255, 0, 0), 5))
imsave(output_filename, draw_edge(input_image, newRidge2, (0, 0, 255), 5))
imsave(output_filename, draw_edge(input_image, humanSample, (0, 255, 0), 5))
