#!/usr/bin/env python

import cv2
import sys
import numpy as np
from random import randint

#---------------------------------------------------------------------------------------#
#|                      Best Fit Patch and related functions                           |#
#---------------------------------------------------------------------------------------#
def OverlapErrorVertical2(imgTarget, imgSample, imgPx, samplePx):
    iLeft,jLeft = imgPx
    iRight,jRight = samplePx
    roiLeft=imgTarget[iLeft:iLeft+patchSize, jLeft:jLeft+OverlapWidth]
    roiRight=imgSample[iRight:iRight+patchSize, jRight:jRight+OverlapWidth]

    diff = (roiLeft.astype(np.int) - roiRight.astype(np.int))**2
    diff = np.sum(np.sqrt(np.sum(diff, -1)))

    return diff

def OverlapErrorHorizntl2( imgTarget, imgSample, leftPx, rightPx ):
    iLeft,jLeft = leftPx
    iRight,jRight = rightPx
    roiLeft=imgTarget[iLeft:iLeft+OverlapWidth, jLeft:jLeft+patchSize]
    roiRight=imgSample[iRight:iRight+OverlapWidth, jRight:jRight+patchSize]

    diff = (roiLeft.astype(np.int) - roiRight.astype(np.int))**2
    diff = np.sum(np.sqrt(np.sum(diff, -1)))

    return diff

def GetBestPatches( px , imgTarget, imgSample, threshold):#Will get called in GrowImage
    PixelList = []
    sample_height, sample_width = imgSample.shape[0:2]

    #check for top layer
    if px[0] == 0:
        for i in range(sample_height - patchSize):
            for j in range(OverlapWidth, sample_width - patchSize ):
                error = OverlapErrorVertical2( imgTarget, imgSample, (px[0], px[1] - OverlapWidth), (i, j - OverlapWidth)  )
                if error  < threshold:
                    PixelList.append((i,j))
                elif error < threshold/2:
                    return [(i,j)]
    #check for leftmost layer
    elif px[1] == 0:
        for i in range(OverlapWidth, sample_height - patchSize ):
            for j in range(sample_width - patchSize):
                error = OverlapErrorHorizntl2( imgTarget, imgSample, (px[0] - OverlapWidth, px[1]), (i - OverlapWidth, j)  )
                if error  < threshold:
                    PixelList.append((i,j))
                elif error < threshold/2:
                    return [(i,j)]
    #for pixel placed inside 
    else:
        for i in range(OverlapWidth, sample_height - patchSize):
            for j in range(OverlapWidth, sample_width - patchSize):
                error_Vertical   = OverlapErrorVertical2( imgTarget, imgSample, (px[0], px[1] - OverlapWidth), (i,j - OverlapWidth)  )
                error_Horizntl   = OverlapErrorHorizntl2( imgTarget, imgSample, (px[0] - OverlapWidth, px[1]), (i - OverlapWidth,j) )
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

def GetCostVertical(imgTarget, imgSample, imgPx, samplePx):
    Cost = np.zeros((patchSize, OverlapWidth))
    for j in range(OverlapWidth):
        for i in range(patchSize):
            if i == patchSize - 1:
                Cost[i,j] = SSD_Error(imgTarget, imgSample, (i ,j - OverlapWidth), imgPx, samplePx)
            else:
                if j == 0 :
                    Cost[i,j] = SSD_Error(imgTarget, imgSample, (i , j - OverlapWidth), imgPx, samplePx) + min( SSD_Error(imgTarget, imgSample, (i + 1, j - OverlapWidth), imgPx, samplePx),SSD_Error(imgTarget, imgSample, (i + 1,j + 1 - OverlapWidth), imgPx, samplePx) )
                elif j == OverlapWidth - 1:
                    Cost[i,j] = SSD_Error(imgTarget, imgSample, (i, j - OverlapWidth), imgPx, samplePx) + min( SSD_Error(imgTarget, imgSample, (i + 1, j - OverlapWidth), imgPx, samplePx), SSD_Error(imgTarget, imgSample, (i + 1, j - 1 - OverlapWidth), imgPx, samplePx) )
                else:
                    Cost[i,j] = SSD_Error(imgTarget, imgSample, (i, j -OverlapWidth), imgPx, samplePx) + min(SSD_Error(imgTarget, imgSample, (i + 1, j - OverlapWidth), imgPx, samplePx),SSD_Error(imgTarget, imgSample, (i + 1, j + 1 - OverlapWidth), imgPx, samplePx),SSD_Error(imgTarget, imgSample, (i + 1, j - 1 - OverlapWidth), imgPx, samplePx))
    return Cost

def GetCostHorizntl(imgTarget, imgSample, imgPx, samplePx):
    Cost = np.zeros((OverlapWidth, patchSize))
    for i in range( OverlapWidth ):
        for j in range( patchSize ):
            if j == patchSize - 1:
                Cost[i,j] = SSD_Error(imgTarget, imgSample, (i - OverlapWidth, j), imgPx, samplePx)
            elif i == 0:
                Cost[i,j] = SSD_Error(imgTarget, imgSample, (i - OverlapWidth, j), imgPx, samplePx) + min(SSD_Error(imgTarget, imgSample, (i - OverlapWidth, j + 1), imgPx, samplePx), SSD_Error(imgTarget, imgSample, (i + 1 - OverlapWidth, j + 1), imgPx, samplePx))
            elif i == OverlapWidth - 1:
                Cost[i,j] = SSD_Error(imgTarget, imgSample, (i - OverlapWidth, j), imgPx, samplePx) + min(SSD_Error(imgTarget, imgSample, (i - OverlapWidth, j + 1), imgPx, samplePx), SSD_Error(imgTarget, imgSample, (i - 1 - OverlapWidth, j + 1), imgPx, samplePx))
            else:
                Cost[i,j] = SSD_Error(imgTarget, imgSample, (i - OverlapWidth, j), imgPx, samplePx) + min(SSD_Error(imgTarget, imgSample, (i - OverlapWidth, j + 1), imgPx, samplePx), SSD_Error(imgTarget, imgSample, (i + 1 - OverlapWidth, j + 1), imgPx, samplePx), SSD_Error(imgTarget, imgSample, (i - 1 - OverlapWidth, j + 1), imgPx, samplePx))
    return Cost

#---------------------------------------------------------------#
#|                  Finding Minimum Cost Path                  |#
#---------------------------------------------------------------#

def FindMinCostPathVertical(Cost):
    Boundary = np.zeros((patchSize),np.int)
    ParentMatrix = np.zeros((patchSize, OverlapWidth),np.int)
    for i in range(1, patchSize):
        for j in range(OverlapWidth):
            if j == 0:
                ParentMatrix[i,j] = j if Cost[i-1,j] < Cost[i-1,j+1] else j+1
            elif j == OverlapWidth - 1:
                ParentMatrix[i,j] = j if Cost[i-1,j] < Cost[i-1,j-1] else j-1
            else:
                curr_min = j if Cost[i-1,j] < Cost[i-1,j-1] else j-1
                ParentMatrix[i,j] = curr_min if Cost[i-1,curr_min] < Cost[i-1,j+1] else j+1
            Cost[i,j] += Cost[i-1, ParentMatrix[i,j]]
    minIndex = 0
    for j in range(1,OverlapWidth):
        minIndex = minIndex if Cost[patchSize - 1, minIndex] < Cost[patchSize - 1, j] else j
    Boundary[patchSize-1] = minIndex
    for i in range(patchSize - 1,0,-1):
        Boundary[i - 1] = ParentMatrix[i,Boundary[i]]
    return Boundary

def FindMinCostPathHorizntl(Cost):
    Boundary = np.zeros(( patchSize),np.int)
    ParentMatrix = np.zeros((OverlapWidth, patchSize),np.int)
    for j in range(1, patchSize):
        for i in range(OverlapWidth):
            if i == 0:
                ParentMatrix[i,j] = i if Cost[i,j-1] < Cost[i+1,j-1] else i + 1
            elif i == OverlapWidth - 1:
                ParentMatrix[i,j] = i if Cost[i,j-1] < Cost[i-1,j-1] else i - 1
            else:
                curr_min = i if Cost[i,j-1] < Cost[i-1,j-1] else i - 1
                ParentMatrix[i,j] = curr_min if Cost[curr_min,j-1] < Cost[i-1,j-1] else i + 1
            Cost[i,j] += Cost[ParentMatrix[i,j], j-1]
    minIndex = 0
    for i in range(1,OverlapWidth):
        minIndex = minIndex if Cost[minIndex, patchSize - 1] < Cost[i, patchSize - 1] else i
    Boundary[patchSize-1] = minIndex
    for j in range(patchSize - 1,0,-1):
        Boundary[j - 1] = ParentMatrix[Boundary[j],j]
    return Boundary

#---------------------------------------------------------------#
#|                      Quilting                               |#
#---------------------------------------------------------------#

def QuiltVertical(Boundary, imgTarget, imgSample, imgPx, samplePx):
    for i in range(patchSize):
        for j in range(Boundary[i], 0, -1):
            imgTarget[imgPx[0] + i, imgPx[1] - j] = imgSample[ samplePx[0] + i, samplePx[1] - j ]
def QuiltHorizntl(Boundary, imgTarget, imgSample, imgPx, samplePx):
    for j in range(patchSize):
        for i in range(Boundary[j], 0, -1):
            imgTarget[imgPx[0] - i, imgPx[1] + j] = imgSample[samplePx[0] - i, samplePx[1] + j]

def QuiltPatches( imgTarget, imgSample, imgPx, samplePx ):
    #check for top layer
    if imgPx[0] == 0:
        Cost = GetCostVertical(imgTarget, imgSample, imgPx, samplePx)
        # Getting boundary to stitch
        Boundary = FindMinCostPathVertical(Cost)
        #Quilting Patches
        QuiltVertical(Boundary, imgTarget, imgSample, imgPx, samplePx)
    #check for leftmost layer
    elif imgPx[1] == 0:
        Cost = GetCostHorizntl(imgTarget, imgSample, imgPx, samplePx)
        #Boundary to stitch
        Boundary = FindMinCostPathHorizntl(Cost)
        #Quilting Patches
        QuiltHorizntl(Boundary, imgTarget, imgSample, imgPx, samplePx)
    #for pixel placed inside 
    else:
        CostVertical = GetCostVertical(imgTarget, imgSample, imgPx, samplePx)
        CostHorizntl = GetCostHorizntl(imgTarget, imgSample, imgPx, samplePx)
        BoundaryVertical = FindMinCostPathVertical(CostVertical)
        BoundaryHorizntl = FindMinCostPathHorizntl(CostHorizntl)
        QuiltVertical(BoundaryVertical, imgTarget, imgSample, imgPx, samplePx)
        QuiltHorizntl(BoundaryHorizntl, imgTarget, imgSample, imgPx, samplePx)

#--------------------------------------------------------------------------------------------------------#
#                                   Growing Image Patch-by-patch                                        |#
#--------------------------------------------------------------------------------------------------------#

def FillImage( imgTarget, imgSample, imgPx, samplePx ):
    imgTarget[imgPx[0]:imgPx[0]+patchSize, imgPx[1]:imgPx[1]+patchSize] = imgSample[samplePx[0]:samplePx[0]+patchSize, samplePx[1]:samplePx[1]+patchSize]


def drawDumpImg(imgTarget, imgSample, bestMatchesList, sampleID, patchSize):
    targetH, targetW = imgTarget.shape[0:2]
    sampleH, sampleW = imgSample.shape[0:2]

    imgSampleClone=imgSample.copy()
    for match in bestMatchesList:
        point1 = (match[1], match[0])
        point2 = (match[1]+patchSize, match[0]+patchSize)
        cv2.rectangle(imgSampleClone, point1, point2, (0, 255, 0), 1, cv2.LINE_AA)

    dumpH = max(targetH, sampleH)
    dumpW = 2*max(targetW, sampleW)

    if len(imgTarget.shape)==2:
        dumpImg = np.zeros((dumpH, dumpW), np.uint8)
    else:
        dumpImg = np.zeros((dumpH, dumpW, 3), np.uint8)

    print("dumpImg.shape", dumpImg.shape)

    top = (dumpH - targetH)//2
    left= (dumpW//2 - targetW)//2
    dumpImg[top:top+targetH, left:left+targetW]=imgTarget
    top = (dumpH - sampleH)//2
    left=dumpW//2 + (dumpW//2-sampleW)//2
    print(top, left, sampleW)

    print(dumpImg[top:top+sampleH, left:left+sampleW].shape)
    dumpImg[top:top+sampleH, left:left+sampleW]=imgSampleClone

    cv2.imwrite("./dumpImg.jpg", dumpImg)

def synthesis(imgSample, patchSize, OverlapWidth, InitialThresConstant):
    img_height = int((user_desired_img_height // patchSize)*patchSize+patchSize)
    img_width = int((user_desired_img_width // patchSize)*patchSize+patchSize)
    sample_width = imgSample.shape[1]
    sample_height = imgSample.shape[0]
    img = np.zeros((img_height,img_width,3), np.uint8)

    #Picking random patch to begin
    randomPatchHeight = randint(0, sample_height - patchSize)
    randomPatchWidth = randint(0, sample_width - patchSize)
    img[0:patchSize,0:patchSize] = imgSample[randomPatchHeight:randomPatchHeight+patchSize, randomPatchWidth:randomPatchWidth+patchSize]
    #initializating next 
    GrowPatchLocation = (0,patchSize)
    drawDumpImg(img, imgSample, list(), 0)

    pixelsCompleted = 0
    TotalPatches = ( (img_height)/ patchSize )*((img_width)/ patchSize) - 1
    sys.stdout.write("Progress : [%-20s] %d%% | PixelsCompleted: %d | ThresholdConstant: --.------" % ('='*(int)(pixelsCompleted*20/TotalPatches), (100*pixelsCompleted)/TotalPatches, pixelsCompleted))
    sys.stdout.flush()
    while pixelsCompleted<TotalPatches:
        ThresholdConstant = InitialThresConstant
        #set progress to zer0
        progress = 0 
        while progress == 0:
            ThresholdOverlapError = ThresholdConstant * patchSize * OverlapWidth
            #Get Best matches for current pixel
            List = GetBestPatches(GrowPatchLocation, img, imgSample, ThresholdOverlapError)
            if len(List) > 0:
                progress = 1
                #Make A random selection from best fit pxls
                sampleMatch = List[ randint(0, len(List) - 1) ]
                FillImage( img, imgSample, GrowPatchLocation, sampleMatch )
                #Quilt this with in curr location
                QuiltPatches( img, imgSample, GrowPatchLocation, sampleMatch )
                #upadate cur pixel location
                GrowPatchLocation = (GrowPatchLocation[0], GrowPatchLocation[1] + patchSize)
                if GrowPatchLocation[1] + patchSize > img_width:
                    GrowPatchLocation = (GrowPatchLocation[0] + patchSize, 0)
            #if not progressed, increse threshold
            else:
                ThresholdConstant *= 1.1
        pixelsCompleted += 1
        # print pixelsCompleted, ThresholdConstant
        sys.stdout.write('\r')
        sys.stdout.write("Progress : [%-20s] %d%% | PixelsCompleted: %d | ThresholdConstant: %f" % ('='*(int)(pixelsCompleted*20/TotalPatches), (100*pixelsCompleted)/TotalPatches, pixelsCompleted, ThresholdConstant))
        sys.stdout.flush()

    img=img[0:user_desired_img_height,0:user_desired_img_width]
    return img

def transfer(imgTarget, imgTexure, patchSize, OverlapWidth, InitialThresConstant):
    pass

    
#Image Loading and initializations
InputName = str(sys.argv[1])
patchSize = int(sys.argv[2])
OverlapWidth = int(sys.argv[3])
InitialThresConstant = float(sys.argv[4])

imgSample = cv2.imread(InputName)
user_desired_img_height = 250
user_desired_img_width = 250

img = synthesis(imgSample, patchSize, OverlapWidth, InitialThresConstant)

# Displaying Images
cv2.imshow('Sample Texture',imgSample)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow('Generated Image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
