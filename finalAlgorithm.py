# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 01:02:07 2013
Started on: 12/26/2013
Last Date:
@author: cmzprobk
"""

import numpy as np
import matplotlib.pyplot as plt
import time

class Stroke:
    def __init__(self):
        self.times = []
        self.rawData = []
        self.deg = 2
        self.range = 20
        self.normData = []
        self.shiftedTime = []
        self.shiftedData = []
        self.startShiftIdx = 0
        self.normTime = []
        self.normData = []
        self.startNormIdx = 0
        self.startXValue = 0
        
    def setHigh(self, high, highIdx):
        self.high = high
        self.highIdx = highIdx
    
    def addNormData(self, normData):
        self.normData = normData
        
    def addRawDataPt(self, time, pt):
        self.rawData.append(pt)
        self.times.append(time)
    
    def normalizeData(self):
        # Find the fit for the highest point
        lo = self.highIdx - self.range
        hi = self.highIdx + self.range
        pHi = np.polyfit(self.times[lo:hi], self.rawData[lo:hi], self.deg)
        
        # Find the max of parabola p_1(x^2) + p_2(x) + p_3
        # The x value of the max = (-p_2)/(2*p_1)
        self.maxX = (-pHi[1])/(2*pHi[0])
        self.maxY = np.polyval(pHi, self.maxX)

        # First scale the points down
        timeMin = self.times[0]
        timeMax = self.times[self.highIdx]
        rawLo = self.rawData[0]
        k = self.startShiftIdx
        while (k < len(self.rawData)):
            self.shiftedTime.append((float)(self.times[k]-timeMin)/timeMax);
            self.shiftedData.append((self.rawData[k] - rawLo)/self.maxY)
            k += 1
        self.startShiftIdx = k
            
        # Divide range of X values by number of points desired
        divide = 70
        xStep =  (float) (self.shiftedTime[self.highIdx] - self.shiftedTime[0])/divide
        xValue = self.startXValue
        j = self.startNormIdx
        sz = len(self.shiftedTime)
        
        # Calculate new interpolated X values
        bound = 15
        pStart = np.polyfit(self.shiftedTime[0:9], self.shiftedData[0:9], self.deg)
        pEnd = np.polyfit(self.shiftedTime[sz-bound*2-1:sz-1], self.shiftedData[sz-bound*2-1:sz-1], self.deg)
        while j < sz:
            # Add the new time x value
            visited = False
            while j < sz and self.shiftedTime[j] < xValue:
                if (j == self.highIdx):
                    self.normDataHigh = len(self.normTime)
                j += 1
            visited = True
            if j < sz and self.shiftedTime[j] == xValue and visited:
                self.normTime.append(xValue)
                self.normData.append(self.shiftedData[j])
                xValue += xStep
            elif j < sz and visited:
                self.normTime.append(xValue)
            # Use a precalculated p if the time is at the start or end
                if (j < bound):
                    self.normData.append(np.polyval(pStart, xValue))
                elif (j > sz-bound-1):
                    self.normData.append(np.polyval(pEnd, xValue))
                else:
                    # Calculate p to find the shifted y
                    p = np.polyfit(self.shiftedTime[j-bound:j+bound], self.shiftedData[j-bound:j+bound], self.deg)
                    self.normData.append(np.polyval(p, xValue))
                xValue += xStep
        
        self.startNormIdx = j
        self.startXValue = xValue

# First comparison method
    def compareStrokes(self, s):
        shortLen = 0
        if (len(self.normTime) < len(s.normTime)):
            shortLen = len(self.normTime)
        else:
            shortLen = len(s.normTime)

        sumDiffs = float(0)
        for i in range(shortLen):
            sumDiffs += float(np.abs(self.normData[i] - s.normData[i])) 
        return float(sumDiffs/shortLen)
        
    def findNearestMatch(self, allStrokes, ignoreIdx):
        minDiff = float("inf")
        minStroke = 0
        # Compare all possible strokes and find the minimum difference
        for k in range(len(allStrokes)):
            #Calculate the difference
            diff = self.compareStrokes(allStrokes[k])
            if (diff < minDiff and k != ignoreIdx):
                minDiff = diff
                minStroke = allStrokes[k]
        return minStroke

# Final Prediction Method
    def predictStrike(self, simStroke):      
        i = simStroke.normDataHigh
        closestX = 0
        closestY = 0
        curX = self.normTime[len(self.normTime)-1]
        curY = self.normData[len(self.normData)-1]
        
        # Use a linear search to find the closest Y point
        while (i < len(simStroke.normData) and simStroke.normData[i] > curY):
            x = simStroke.normTime[i]
            y = simStroke.normData[i]
            if (np.abs(curY - y) < np.abs(curY - closestY)):
                closestY = y
                closestX = x
            i += 1
        bound = 15
        if (i < len(simStroke.normData)- bound - 1):
            p = np.polyfit(simStroke.normData[i-bound-1:i+bound], simStroke.normTime[i-bound-1:i+bound], simStroke.deg)
            closestX = np.polyval(p, simStroke.normData[i-1])
        else:
            sz = len(simStroke.normTime)
            pEnd = np.polyfit(simStroke.normData[sz-bound-1:sz-1], simStroke.normTime[sz-bound-1:sz-1], simStroke.deg)
            closestX = np.polyval(pEnd, simStroke.normData[i-1])
            
        simTime = float(simStroke.normTime[len(simStroke.normTime)-1])
        if (closestX != 0):
            normStrikeTime = float(simTime*curX/closestX)
            return float((normStrikeTime*self.times[self.highIdx]) + self.times[0])

        else:
            percentToHi = float(simStroke.highIdx)/float(len(simStroke.rawData))
            timeToHi = float(self.times[self.highIdx])
            return float(timeToHi/percentToHi)
      
# The data processing part of the algorithm     
# Open the file with the data
start_time = time.time()
file = open("all_notes", "r")

# The points will ve stored in the points array
points = list()
i = 0;
topFound = False
firstFound = False
prevPt = 0
prevPrevPt = 0
high = 0
highIdx = 0

# Holds a list of all strokes
allStrokes = []  
s = Stroke()

# Read through every point in the file (each is in a new line)
for line in file: 
    # Convert the string value to the integer
    pt = int(line)
    s.addRawDataPt(i*2, pt)

    # If the end of a stroke is found
    if (i > 5) and (topFound is True) and (pt > prevPt) and (pt > prevPrevPt):
        pt = 0
        if (firstFound is False):
            firstFound = True
        else:            
            # Save the information for the stroke
            s.normalizeData()
            allStrokes.append(s)
            
        # Reset the variables used            
        topFound = False
        i = 0
        high = 0
        highIdx = 0
        numPts = 0
        points = []
        rawData = []
        times = []
        s = Stroke()
        
    # Means the top of the stroke is found
    elif (i != 0) and (topFound is False) and (pt < prevPt) & (pt > 20):
        topFound = True
        high = pt
        highIdx = i 
        s.setHigh(high, highIdx)
    
    # Update the variables
    i += 1
    prevPrevPt = prevPt
    prevPt = pt
    
# Make predictions based on all strokes
dataFile = open('data.txt', 'w')
totalDiff = 0
totalDiffAbs = 0
NewThirtyDiffAbs = 0
ThirtyDiffAbs = 0
TwentyDiffAbs = 0
TenDiffAbs = 0
NewThirtyFound = False
ThirtyFound = False
TwentyFound = False
TenFound = False

diff = []
timeLeft = 0
for curI in range(len(allStrokes)):
    NewThirtyFound = False
    ThirtyFound = False
    TwentyFound = False
    TenFound = False
    curStroke = allStrokes[curI]
    rebuiltStroke = Stroke()
    rebuiltStroke.setHigh(curStroke.high, curStroke.highIdx)
    for l in range(len(curStroke.rawData)):
        rebuiltStroke.addRawDataPt(l*2, curStroke.rawData[l])
        if (l > curStroke.highIdx):
            rebuiltStroke.normalizeData()
            simStroke = rebuiltStroke.findNearestMatch(allStrokes, curI)
            predTime = rebuiltStroke.predictStrike(simStroke)

            # See how accurate predictions are 30, 20, and 10 ms out from strike
            if (np.abs(predTime - l*2 - 33) < 3 and NewThirtyFound is False):
                NewThirtyDiffAbs += np.abs(predTime - curStroke.times[len(curStroke.times)-1])
                NewThirtyFound = True
                timeLeft += np.abs(predTime - l*2)
                # Calculate the time the message would most likely be received
                receiveTime = l*2 + 20
                if (predTime >= receiveTime):
                    diff.append(predTime - curStroke.times[len(curStroke.times)-1])
                else:
                    diff.append(receiveTime -curStroke.times[len(curStroke.times)-1])
                
            if (predTime - l*2 < 30 and ThirtyFound is False):
                ThirtyDiffAbs += np.abs(predTime - curStroke.times[len(curStroke.times)-1])
                ThirtyFound = True
                #diff.append(predTime - curStroke.times[len(curStroke.times)-1])
            if (predTime - l*2 < 20 and TwentyFound is False):
                TwentyDiffAbs += np.abs(predTime - curStroke.times[len(curStroke.times)-1])
                TwentyFound = True
            if (predTime - l*2 < 10 and TenFound is False):
                TenDiffAbs += np.abs(predTime - curStroke.times[len(curStroke.times)-1])
                TenFound = True
    totalDiff += predTime - curStroke.times[len(curStroke.times)-1]
    totalDiffAbs += np.abs(predTime - curStroke.times[len(curStroke.times)-1])
    
    for k in range(len(rebuiltStroke.normData)):
        #print str(rebuiltStroke.normData[k]) + " " + str(curStroke.normData[k])
        z = 10
print "Time Left: " + str(float(timeLeft/len(allStrokes)))
print "New 30 ms prediction: " + str(float(NewThirtyDiffAbs/len(allStrokes)))
dataFile.write("AVG DIFFERENCE: " + str((float)(totalDiff / len(allStrokes))) + "\n")
dataFile.write("AVG ABS DIFFERENCE: " + str((float)(totalDiffAbs / len(allStrokes))) + "\n")
dataFile.write("AVG ABS DIFF 30 ms OUT: " + str((float)(ThirtyDiffAbs / len(allStrokes))) + "\n")
dataFile.write("AVG ABS DIFF 20 ms OUT: " + str((float)(TwentyDiffAbs / len(allStrokes))) + "\n")
dataFile.write("AVG ABS DIFF 10 ms OUT: " + str((float)(TenDiffAbs / len(allStrokes))) + "\n")
dataFile.close()
print "Done!!"

plt.hist(diff, 30)
plt.xlabel("Average absolute difference: 30 ms before predicted strike")
plt.ylabel("Number of strokes in bin")
plt.show()
print time.time() - start_time, "seconds"