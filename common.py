#!/usr/bin/env python

import cv2
import sys
import numpy as np
from random import randint
import os
#for gif making
import imageio 
from PIL import Image
from math import floor

class Params:
    def __init__(self, patchSize, overlapWidth, initialThresConstant, userDesiredSize):
        self.patchSize = patchSize
        self.overlapWidth = overlapWidth
        self.initialThresConstant = initialThresConstant
        self.userDesiredSize = userDesiredSize

#---------------------------------------------------------------------------------------#
#|                      Best Fit Patch and related functions                           |#
#---------------------------------------------------------------------------------------#
def OverlapErrorVertical2(imgTarget, imgSample, imgPx, samplePx, params):
    iLeft,jLeft = imgPx
    iRight,jRight = samplePx
    roiLeft=imgTarget[iLeft:iLeft+params.patchSize, jLeft:jLeft+params.overlapWidth]
    roiRight=imgSample[iRight:iRight+params.patchSize, jRight:jRight+params.overlapWidth]

    diff = (roiLeft.astype(np.int) - roiRight.astype(np.int))**2
    diff = np.sum(np.sqrt(np.sum(diff, -1)))

    return diff

def OverlapErrorHorizntl2( imgTarget, imgSample, leftPx, rightPx, params):
    iLeft,jLeft = leftPx
    iRight,jRight = rightPx
    roiLeft=imgTarget[iLeft:iLeft+params.overlapWidth, jLeft:jLeft+params.patchSize]
    roiRight=imgSample[iRight:iRight+params.overlapWidth, jRight:jRight+params.patchSize]

    diff = (roiLeft.astype(np.int) - roiRight.astype(np.int))**2
    diff = np.sum(np.sqrt(np.sum(diff, -1)))

    return diff

def GetBestPatches( px , imgTarget, imgSample, threshold, params):#Will get called in GrowImage
    PixelList = []
    sample_height, sample_width = imgSample.shape[0:2]

    #check for top layer
    if px[0] == 0:
        for i in range(sample_height - patchSize):
            for j in range(overlapWidth, sample_width - patchSize ):
                error = OverlapErrorVertical2( imgTarget, imgSample, (px[0], px[1] - overlapWidth), (i, j - overlapWidth)  )
                if error  < threshold:
                    PixelList.append((i,j))
                elif error < threshold/2:
                    return [(i,j)]
    #check for leftmost layer
    elif px[1] == 0:
        for i in range(overlapWidth, sample_height - patchSize ):
            for j in range(sample_width - patchSize):
                error = OverlapErrorHorizntl2( imgTarget, imgSample, (px[0] - overlapWidth, px[1]), (i - overlapWidth, j)  )
                if error  < threshold:
                    PixelList.append((i,j))
                elif error < threshold/2:
                    return [(i,j)]
    #for pixel placed inside 
    else:
        for i in range(overlapWidth, sample_height - patchSize):
            for j in range(overlapWidth, sample_width - patchSize):
                error_Vertical   = OverlapErrorVertical2( imgTarget, imgSample, (px[0], px[1] - overlapWidth), (i,j - overlapWidth)  )
                error_Horizntl   = OverlapErrorHorizntl2( imgTarget, imgSample, (px[0] - overlapWidth, px[1]), (i - overlapWidth,j) )
                if error_Vertical  < threshold and error_Horizntl < threshold:
                    PixelList.append((i,j))
                elif error_Vertical < threshold/2 and error_Horizntl < threshold/2:
                    return [(i,j)]
    return PixelList

#-----------------------------------------------------------------------------------------------#
#|                              Quilting and related Functions                                 |#
#-----------------------------------------------------------------------------------------------#

def SSD_Error( imgTarget, imgSample, offset, imgPx, samplePx ):
    imgValue = imgTarget[imgPx[0]+offset[0], imgPx[1]+offset[1]]
    sampleValue = imgSample[samplePx[0] + offset[0], samplePx[1] + offset[1]]
    
    diff = imgValue.astype(int) - sampleValue.astype(int)
    diff = np.mean(diff**2)
    return diff

#---------------------------------------------------------------#
#|                  Calculating Cost                           |#
#---------------------------------------------------------------#

def GetCostVertical(imgTarget, imgSample, imgPx, samplePx, params):
    overlapWidth = params.overlapWidth
    patchSize = params.patchSize
    Cost = np.zeros((patchSize, overlapWidth))
    for j in range(overlapWidth):
        for i in range(patchSize):
            if i == patchSize - 1:
                Cost[i,j] = SSD_Error(imgTarget, imgSample, (i ,j - overlapWidth), imgPx, samplePx)
            else:
                if j == 0 :
                    Cost[i,j] = SSD_Error(imgTarget, imgSample, (i , j - overlapWidth), imgPx, samplePx) + min( SSD_Error(imgTarget, imgSample, (i + 1, j - overlapWidth), imgPx, samplePx),SSD_Error(imgTarget, imgSample, (i + 1,j + 1 - overlapWidth), imgPx, samplePx) )
                elif j == overlapWidth - 1:
                    Cost[i,j] = SSD_Error(imgTarget, imgSample, (i, j - overlapWidth), imgPx, samplePx) + min( SSD_Error(imgTarget, imgSample, (i + 1, j - overlapWidth), imgPx, samplePx), SSD_Error(imgTarget, imgSample, (i + 1, j - 1 - overlapWidth), imgPx, samplePx) )
                else:
                    Cost[i,j] = SSD_Error(imgTarget, imgSample, (i, j -overlapWidth), imgPx, samplePx) + min(SSD_Error(imgTarget, imgSample, (i + 1, j - overlapWidth), imgPx, samplePx),SSD_Error(imgTarget, imgSample, (i + 1, j + 1 - overlapWidth), imgPx, samplePx),SSD_Error(imgTarget, imgSample, (i + 1, j - 1 - overlapWidth), imgPx, samplePx))
    return Cost

def GetCostHorizntl(imgTarget, imgSample, imgPx, samplePx, params):
    overlapWidth = params.overlapWidth
    patchSize = params.patchSize

    Cost = np.zeros((overlapWidth, patchSize))
    for i in range( overlapWidth ):
        for j in range( patchSize ):
            if j == patchSize - 1:
                Cost[i,j] = SSD_Error(imgTarget, imgSample, (i - overlapWidth, j), imgPx, samplePx)
            elif i == 0:
                Cost[i,j] = SSD_Error(imgTarget, imgSample, (i - overlapWidth, j), imgPx, samplePx) + min(SSD_Error(imgTarget, imgSample, (i - overlapWidth, j + 1), imgPx, samplePx), SSD_Error(imgTarget, imgSample, (i + 1 - overlapWidth, j + 1), imgPx, samplePx))
            elif i == overlapWidth - 1:
                Cost[i,j] = SSD_Error(imgTarget, imgSample, (i - overlapWidth, j), imgPx, samplePx) + min(SSD_Error(imgTarget, imgSample, (i - overlapWidth, j + 1), imgPx, samplePx), SSD_Error(imgTarget, imgSample, (i - 1 - overlapWidth, j + 1), imgPx, samplePx))
            else:
                Cost[i,j] = SSD_Error(imgTarget, imgSample, (i - overlapWidth, j), imgPx, samplePx) + min(SSD_Error(imgTarget, imgSample, (i - overlapWidth, j + 1), imgPx, samplePx), SSD_Error(imgTarget, imgSample, (i + 1 - overlapWidth, j + 1), imgPx, samplePx), SSD_Error(imgTarget, imgSample, (i - 1 - overlapWidth, j + 1), imgPx, samplePx))
    return Cost

#---------------------------------------------------------------#
#|                  Finding Minimum Cost Path                  |#
#---------------------------------------------------------------#

def FindMinCostPathVertical(Cost, params):
    patchSize = params.patchSize
    overlapWidth = params.overlapWidth

    Boundary = np.zeros((patchSize),np.int)
    ParentMatrix = np.zeros((patchSize, overlapWidth),np.int)
    for i in range(1, patchSize):
        for j in range(overlapWidth):
            if j == 0:
                ParentMatrix[i,j] = j if Cost[i-1,j] < Cost[i-1,j+1] else j+1
            elif j == overlapWidth - 1:
                ParentMatrix[i,j] = j if Cost[i-1,j] < Cost[i-1,j-1] else j-1
            else:
                curr_min = j if Cost[i-1,j] < Cost[i-1,j-1] else j-1
                ParentMatrix[i,j] = curr_min if Cost[i-1,curr_min] < Cost[i-1,j+1] else j+1
            Cost[i,j] += Cost[i-1, ParentMatrix[i,j]]
    minIndex = 0
    for j in range(1,overlapWidth):
        minIndex = minIndex if Cost[patchSize - 1, minIndex] < Cost[patchSize - 1, j] else j
    Boundary[patchSize-1] = minIndex
    for i in range(patchSize - 1,0,-1):
        Boundary[i - 1] = ParentMatrix[i,Boundary[i]]
    return Boundary

def FindMinCostPathHorizntl(Cost, params):
    patchSize = params.patchSize
    overlapWidth = params.overlapWidth

    Boundary = np.zeros(( patchSize),np.int)
    ParentMatrix = np.zeros((overlapWidth, patchSize),np.int)
    for j in range(1, patchSize):
        for i in range(overlapWidth):
            if i == 0:
                ParentMatrix[i,j] = i if Cost[i,j-1] < Cost[i+1,j-1] else i + 1
            elif i == overlapWidth - 1:
                ParentMatrix[i,j] = i if Cost[i,j-1] < Cost[i-1,j-1] else i - 1
            else:
                curr_min = i if Cost[i,j-1] < Cost[i-1,j-1] else i - 1
                ParentMatrix[i,j] = curr_min if Cost[curr_min,j-1] < Cost[i-1,j-1] else i + 1
            Cost[i,j] += Cost[ParentMatrix[i,j], j-1]
    minIndex = 0
    for i in range(1,overlapWidth):
        minIndex = minIndex if Cost[minIndex, patchSize - 1] < Cost[i, patchSize - 1] else i
    Boundary[patchSize-1] = minIndex
    for j in range(patchSize - 1,0,-1):
        Boundary[j - 1] = ParentMatrix[Boundary[j],j]
    return Boundary

#---------------------------------------------------------------#
#|                      Quilting                               |#
#---------------------------------------------------------------#

def QuiltVertical(Boundary, imgTarget, imgSample, imgPx, samplePx, params):
    for i in range(params.patchSize):
        for j in range(Boundary[i], 0, -1):
            imgTarget[imgPx[0] + i, imgPx[1] - j] = imgSample[ samplePx[0] + i, samplePx[1] - j ]
def QuiltHorizntl(Boundary, imgTarget, imgSample, imgPx, samplePx, params):
    for j in range(params.patchSize):
        for i in range(Boundary[j], 0, -1):
            imgTarget[imgPx[0] - i, imgPx[1] + j] = imgSample[samplePx[0] - i, samplePx[1] + j]

def QuiltPatches( imgTarget, imgSample, imgPx, samplePx, params):
    #check for top layer
    if imgPx[0] == 0:
        Cost = GetCostVertical(imgTarget, imgSample, imgPx, samplePx, params)
        # Getting boundary to stitch
        Boundary = FindMinCostPathVertical(Cost, params)
        #Quilting Patches
        QuiltVertical(Boundary, imgTarget, imgSample, imgPx, samplePx, params)
    #check for leftmost layer
    elif imgPx[1] == 0:
        Cost = GetCostHorizntl(imgTarget, imgSample, imgPx, samplePx, params)
        #Boundary to stitch
        Boundary = FindMinCostPathHorizntl(Cost, params)
        #Quilting Patches
        QuiltHorizntl(Boundary, imgTarget, imgSample, imgPx, samplePx, params)
    #for pixel placed inside 
    else:
        CostVertical = GetCostVertical(imgTarget, imgSample, imgPx, samplePx, params)
        CostHorizntl = GetCostHorizntl(imgTarget, imgSample, imgPx, samplePx, params)
        BoundaryVertical = FindMinCostPathVertical(CostVertical, params)
        BoundaryHorizntl = FindMinCostPathHorizntl(CostHorizntl, params)
        QuiltVertical(BoundaryVertical, imgTarget, imgSample, imgPx, samplePx, params)
        QuiltHorizntl(BoundaryHorizntl, imgTarget, imgSample, imgPx, samplePx, params)

#--------------------------------------------------------------------------------------------------------#
#                                   Growing Image Patch-by-patch                                        |#
#--------------------------------------------------------------------------------------------------------#

def FillImage( imgTarget, imgSample, imgPx, samplePx, params):
    patchSize = params.patchSize
    imgTarget[imgPx[0]:imgPx[0]+patchSize, imgPx[1]:imgPx[1]+patchSize] = imgSample[samplePx[0]:samplePx[0]+patchSize, samplePx[1]:samplePx[1]+patchSize]


NEED_DUMP = True
def makeGif(dumpPath, outputPath, frame_every_X_steps = 15, repeat_ending = 15):
    number_files = len(os.listdir(dumpPath))-2
    frame_every_X_steps = frame_every_X_steps
    repeat_ending = repeat_ending
    steps = np.arange(floor(number_files/frame_every_X_steps)) * frame_every_X_steps
    steps = steps + (number_files - np.max(steps))

    images = []
    for f in steps:
        filename = dumpPath + 'dumpImg%03d.jpg'%f
        images.append(imageio.imread(filename))

    #repeat ending
    for _ in range(repeat_ending):
        filename = dumpPath + 'dumpImg%03d.jpg'%number_files
        images.append(imageio.imread(filename))  
        
    imageio.mimsave(outputPath, images)

def drawDumpImg(imgTarget, imgSample, GrowPatchLocation, bestMatchesList, bestMatch, patchSize, overlapWidth, dumpID):
    if NEED_DUMP==False:
        return

    targetH, targetW = imgTarget.shape[0:2]
    sampleH, sampleW = imgSample.shape[0:2]

    imgSampleClone=imgSample.copy()
    for match in bestMatchesList:
        point1 = (match[1], match[0])
        point2 = (match[1]+patchSize, match[0]+patchSize)
        cv2.rectangle(imgSampleClone, point1, point2, (0, 255, 0), 1, cv2.LINE_8)
    
    if len(bestMatchesList)>0:
        point1 = (bestMatch[1], bestMatch[0])
        point2 = (point1[0]+patchSize, point1[1 ]+patchSize)
        cv2.rectangle(imgSampleClone, point1, point2, (0, 0, 255), 2, cv2.LINE_8)

        if GrowPatchLocation[1]>=patchSize:
            point1 = (bestMatch[1]-overlapWidth, bestMatch[0])
            point2 = (bestMatch[1], bestMatch[0]+patchSize)
            cv2.rectangle(imgSampleClone, point1, point2, (255, 255, 255), 2, cv2.LINE_8)

        if GrowPatchLocation[0]>=patchSize:
            point1 = (bestMatch[1], bestMatch[0]-overlapWidth)
            point2 = (point1[0]+patchSize, point1[1]+overlapWidth)
            cv2.rectangle(imgSampleClone, point1, point2, (255, 255, 255), 2, cv2.LINE_8)


    imgTargetClone = imgTarget.copy()
    point1 = (GrowPatchLocation[1], GrowPatchLocation[0])
    point2 = (point1[0]+patchSize, point1[1 ]+patchSize)
    cv2.rectangle(imgTargetClone, point1, point2, (0, 255, 255), 2, cv2.LINE_8)

    if GrowPatchLocation[1]>=patchSize:
        point1 = (point1[0]-overlapWidth, point1[1])
        point2 = (point1[0]+overlapWidth, point1[1]+patchSize)
        cv2.rectangle(imgTargetClone, point1, point2, (255, 255, 255), 2, cv2.LINE_8)
    
    if GrowPatchLocation[0]>=patchSize:
        point1 = (GrowPatchLocation[1], GrowPatchLocation[0]-overlapWidth)
        point2 = (point1[0]+patchSize, point1[1]+overlapWidth)
        cv2.rectangle(imgTargetClone, point1, point2, (255, 255, 255), 2, cv2.LINE_8)



    dumpH = max(targetH, sampleH)
    dumpW = 2*max(targetW, sampleW)

    if len(imgTargetClone.shape)==2:
        dumpImg = np.zeros((dumpH, dumpW), np.uint8)
    else:
        dumpImg = np.zeros((dumpH, dumpW, 3), np.uint8)

    top = (dumpH - targetH)//2
    left= (dumpW//2 - targetW)//2
    dumpImg[top:top+targetH, left:left+targetW]=imgTargetClone
    
    top = (dumpH - sampleH)//2
    left=dumpW//2 + (dumpW//2-sampleW)//2
    print(top, left, sampleW)

    print(dumpImg[top:top+sampleH, left:left+sampleW].shape)
    dumpImg[top:top+sampleH, left:left+sampleW]=imgSampleClone

    os.makedirs("./dump", exist_ok=True)
    cv2.imwrite("./dump/dumpImg%03d.jpg"%dumpID, dumpImg)

