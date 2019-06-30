#!/usr/bin/python

import cv2
import sys
import numpy as np
from random import randint

#---------------------------------------------------------------------------------------#
#|                      Best Fit Patch and related functions                           |#
#---------------------------------------------------------------------------------------#
def OverlapErrorVertical2(imgPx, samplePx):
    iLeft,jLeft = imgPx
    iRight,jRight = samplePx
    roiLeft=img[iLeft:iLeft+PatchSize, jLeft:jLeft+OverlapWidth]
    roiRight=img_sample[iRight:iRight+PatchSize, jRight:jRight+OverlapWidth]

    diff = (roiLeft.astype(np.int) - roiRight.astype(np.int))**2
    diff = np.sum(np.sqrt(np.sum(diff, -1)))

    return diff

def OverlapErrorHorizntl2( leftPx, rightPx ):
    iLeft,jLeft = leftPx
    iRight,jRight = rightPx
    roiLeft=img[iLeft:iLeft+OverlapWidth, jLeft:jLeft+PatchSize]
    roiRight=img_sample[iRight:iRight+OverlapWidth, jRight:jRight+PatchSize]

    diff = (roiLeft.astype(np.int) - roiRight.astype(np.int))**2
    diff = np.sum(np.sqrt(np.sum(diff, -1)))

    return diff

def GetBestPatches( px ):#Will get called in GrowImage
    PixelList = []
    #check for top layer
    if px[0] == 0:
        for i in range(sample_height - PatchSize):
            for j in range(OverlapWidth, sample_width - PatchSize ):
                error = OverlapErrorVertical2( (px[0], px[1] - OverlapWidth), (i, j - OverlapWidth)  )
                if error  < ThresholdOverlapError:
                    PixelList.append((i,j))
                elif error < ThresholdOverlapError/2:
                    return [(i,j)]
    #check for leftmost layer
    elif px[1] == 0:
        for i in range(OverlapWidth, sample_height - PatchSize ):
            for j in range(sample_width - PatchSize):
                error = OverlapErrorHorizntl2( (px[0] - OverlapWidth, px[1]), (i - OverlapWidth, j)  )
                if error  < ThresholdOverlapError:
                    PixelList.append((i,j))
                elif error < ThresholdOverlapError/2:
                    return [(i,j)]
    #for pixel placed inside 
    else:
        for i in range(OverlapWidth, sample_height - PatchSize):
            for j in range(OverlapWidth, sample_width - PatchSize):
                error_Vertical   = OverlapErrorVertical2( (px[0], px[1] - OverlapWidth), (i,j - OverlapWidth)  )
                error_Horizntl   = OverlapErrorHorizntl2( (px[0] - OverlapWidth, px[1]), (i - OverlapWidth,j) )
                if error_Vertical  < ThresholdOverlapError and error_Horizntl < ThresholdOverlapError:
                    PixelList.append((i,j))
                elif error_Vertical < ThresholdOverlapError/2 and error_Horizntl < ThresholdOverlapError/2:
                    return [(i,j)]
    return PixelList

#-----------------------------------------------------------------------------------------------#
#|                              Quilting and related Functions                                 |#
#-----------------------------------------------------------------------------------------------#

def SSD_Error( offset, imgPx, samplePx ):
    imgValue = img[imgPx[0]+offset[0], imgPx[1]+offset[1]]
    sampleValue = img_sample[samplePx[0] + offset[0], samplePx[1] + offset[1]]
    
    diff = imgValue.astype(int) - sampleValue.astype(int)
    diff = np.mean(diff**2)
    return diff

#---------------------------------------------------------------#
#|                  Calculating Cost                           |#
#---------------------------------------------------------------#

def GetCostVertical(imgPx, samplePx):
    Cost = np.zeros((PatchSize, OverlapWidth))
    for j in range(OverlapWidth):
        for i in range(PatchSize):
            if i == PatchSize - 1:
                Cost[i,j] = SSD_Error((i ,j - OverlapWidth), imgPx, samplePx)
            else:
                if j == 0 :
                    Cost[i,j] = SSD_Error((i , j - OverlapWidth), imgPx, samplePx) + min( SSD_Error((i + 1, j - OverlapWidth), imgPx, samplePx),SSD_Error((i + 1,j + 1 - OverlapWidth), imgPx, samplePx) )
                elif j == OverlapWidth - 1:
                    Cost[i,j] = SSD_Error((i, j - OverlapWidth), imgPx, samplePx) + min( SSD_Error((i + 1, j - OverlapWidth), imgPx, samplePx), SSD_Error((i + 1, j - 1 - OverlapWidth), imgPx, samplePx) )
                else:
                    Cost[i,j] = SSD_Error((i, j -OverlapWidth), imgPx, samplePx) + min(SSD_Error((i + 1, j - OverlapWidth), imgPx, samplePx),SSD_Error((i + 1, j + 1 - OverlapWidth), imgPx, samplePx),SSD_Error((i + 1, j - 1 - OverlapWidth), imgPx, samplePx))
    return Cost

def GetCostHorizntl(imgPx, samplePx):
    Cost = np.zeros((OverlapWidth, PatchSize))
    for i in range( OverlapWidth ):
        for j in range( PatchSize ):
            if j == PatchSize - 1:
                Cost[i,j] = SSD_Error((i - OverlapWidth, j), imgPx, samplePx)
            elif i == 0:
                Cost[i,j] = SSD_Error((i - OverlapWidth, j), imgPx, samplePx) + min(SSD_Error((i - OverlapWidth, j + 1), imgPx, samplePx), SSD_Error((i + 1 - OverlapWidth, j + 1), imgPx, samplePx))
            elif i == OverlapWidth - 1:
                Cost[i,j] = SSD_Error((i - OverlapWidth, j), imgPx, samplePx) + min(SSD_Error((i - OverlapWidth, j + 1), imgPx, samplePx), SSD_Error((i - 1 - OverlapWidth, j + 1), imgPx, samplePx))
            else:
                Cost[i,j] = SSD_Error((i - OverlapWidth, j), imgPx, samplePx) + min(SSD_Error((i - OverlapWidth, j + 1), imgPx, samplePx), SSD_Error((i + 1 - OverlapWidth, j + 1), imgPx, samplePx), SSD_Error((i - 1 - OverlapWidth, j + 1), imgPx, samplePx))
    return Cost

#---------------------------------------------------------------#
#|                  Finding Minimum Cost Path                  |#
#---------------------------------------------------------------#

def FindMinCostPathVertical(Cost):
    Boundary = np.zeros((PatchSize),np.int)
    ParentMatrix = np.zeros((PatchSize, OverlapWidth),np.int)
    for i in range(1, PatchSize):
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
        minIndex = minIndex if Cost[PatchSize - 1, minIndex] < Cost[PatchSize - 1, j] else j
    Boundary[PatchSize-1] = minIndex
    for i in range(PatchSize - 1,0,-1):
        Boundary[i - 1] = ParentMatrix[i,Boundary[i]]
    return Boundary

def FindMinCostPathHorizntl(Cost):
    Boundary = np.zeros(( PatchSize),np.int)
    ParentMatrix = np.zeros((OverlapWidth, PatchSize),np.int)
    for j in range(1, PatchSize):
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
        minIndex = minIndex if Cost[minIndex, PatchSize - 1] < Cost[i, PatchSize - 1] else i
    Boundary[PatchSize-1] = minIndex
    for j in range(PatchSize - 1,0,-1):
        Boundary[j - 1] = ParentMatrix[Boundary[j],j]
    return Boundary

#---------------------------------------------------------------#
#|                      Quilting                               |#
#---------------------------------------------------------------#

def QuiltVertical(Boundary, imgPx, samplePx):
    for i in range(PatchSize):
        for j in range(Boundary[i], 0, -1):
            img[imgPx[0] + i, imgPx[1] - j] = img_sample[ samplePx[0] + i, samplePx[1] - j ]
def QuiltHorizntl(Boundary, imgPx, samplePx):
    for j in range(PatchSize):
        for i in range(Boundary[j], 0, -1):
            img[imgPx[0] - i, imgPx[1] + j] = img_sample[samplePx[0] - i, samplePx[1] + j]

def QuiltPatches( imgPx, samplePx ):
    #check for top layer
    if imgPx[0] == 0:
        Cost = GetCostVertical(imgPx, samplePx)
        # Getting boundary to stitch
        Boundary = FindMinCostPathVertical(Cost)
        #Quilting Patches
        QuiltVertical(Boundary, imgPx, samplePx)
    #check for leftmost layer
    elif imgPx[1] == 0:
        Cost = GetCostHorizntl(imgPx, samplePx)
        #Boundary to stitch
        Boundary = FindMinCostPathHorizntl(Cost)
        #Quilting Patches
        QuiltHorizntl(Boundary, imgPx, samplePx)
    #for pixel placed inside 
    else:
        CostVertical = GetCostVertical(imgPx, samplePx)
        CostHorizntl = GetCostHorizntl(imgPx, samplePx)
        BoundaryVertical = FindMinCostPathVertical(CostVertical)
        BoundaryHorizntl = FindMinCostPathHorizntl(CostHorizntl)
        QuiltVertical(BoundaryVertical, imgPx, samplePx)
        QuiltHorizntl(BoundaryHorizntl, imgPx, samplePx)

#--------------------------------------------------------------------------------------------------------#
#                                   Growing Image Patch-by-patch                                        |#
#--------------------------------------------------------------------------------------------------------#

def FillImage( imgPx, samplePx ):
    img[imgPx[0]:imgPx[0]+PatchSize, imgPx[1]:imgPx[1]+PatchSize] = img_sample[samplePx[0]:samplePx[0]+PatchSize, samplePx[1]:samplePx[1]+PatchSize]

#Image Loading and initializations
InputName = str(sys.argv[1])
PatchSize = int(sys.argv[2])
OverlapWidth = int(sys.argv[3])
InitialThresConstant = float(sys.argv[4])

img_sample = cv2.imread(InputName)
user_desired_img_height = 250
user_desired_img_width = 250
img_height = int((user_desired_img_height // PatchSize)*PatchSize+PatchSize)
img_width = int((user_desired_img_width // PatchSize)*PatchSize+PatchSize)
sample_width = img_sample.shape[1]
sample_height = img_sample.shape[0]
img = np.zeros((img_height,img_width,3), np.uint8)

#Picking random patch to begin
randomPatchHeight = randint(0, sample_height - PatchSize)
randomPatchWidth = randint(0, sample_width - PatchSize)
img[0:PatchSize,0:PatchSize] = img_sample[randomPatchHeight:randomPatchHeight+PatchSize, randomPatchWidth:randomPatchWidth+PatchSize]
#initializating next 
GrowPatchLocation = (0,PatchSize)

pixelsCompleted = 0
TotalPatches = ( (img_height)/ PatchSize )*((img_width)/ PatchSize) - 1
sys.stdout.write("Progress : [%-20s] %d%% | PixelsCompleted: %d | ThresholdConstant: --.------" % ('='*(int)(pixelsCompleted*20/TotalPatches), (100*pixelsCompleted)/TotalPatches, pixelsCompleted))
sys.stdout.flush()
while pixelsCompleted<TotalPatches:
    ThresholdConstant = InitialThresConstant
    #set progress to zer0
    progress = 0 
    while progress == 0:
        ThresholdOverlapError = ThresholdConstant * PatchSize * OverlapWidth
        #Get Best matches for current pixel
        List = GetBestPatches(GrowPatchLocation)
        if len(List) > 0:
            progress = 1
            #Make A random selection from best fit pxls
            sampleMatch = List[ randint(0, len(List) - 1) ]
            FillImage( GrowPatchLocation, sampleMatch )
            #Quilt this with in curr location
            QuiltPatches( GrowPatchLocation, sampleMatch )
            #upadate cur pixel location
            GrowPatchLocation = (GrowPatchLocation[0], GrowPatchLocation[1] + PatchSize)
            if GrowPatchLocation[1] + PatchSize > img_width:
                GrowPatchLocation = (GrowPatchLocation[0] + PatchSize, 0)
        #if not progressed, increse threshold
        else:
            ThresholdConstant *= 1.1
    pixelsCompleted += 1
    # print pixelsCompleted, ThresholdConstant
    sys.stdout.write('\r')
    sys.stdout.write("Progress : [%-20s] %d%% | PixelsCompleted: %d | ThresholdConstant: %f" % ('='*(int)(pixelsCompleted*20/TotalPatches), (100*pixelsCompleted)/TotalPatches, pixelsCompleted, ThresholdConstant))
    sys.stdout.flush()

img=img[0:user_desired_img_height,0:user_desired_img_width]
    
# Displaying Images
cv2.imshow('Sample Texture',img_sample)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow('Generated Image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
