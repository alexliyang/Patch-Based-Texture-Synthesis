#!/usr/bin/env python

from common import *

#---------------------------------------------------------------------------------------#
#|                      Best Fit Patch and related functions                           |#
#---------------------------------------------------------------------------------------#
def GetBestPatches( px , imgTarget, imgTexture, threshold, params):#Will get called in GrowImage
    PixelbestPatchList = []
    sample_height, sample_width = imgTexture.shape[0:2]
    targetROI = imgTarget[px[0]:px[0]+params.patchSize, px[1]:px[1]+params.patchSize]
    #check for top layer
    if px[0] == 0:
        errorArray = (1e+9)*np.ones((sample_height, sample_width))
        for i in range(sample_height - params.patchSize):
            for j in range(params.overlapWidth, sample_width - params.patchSize ):
                textureROI=imgTexture[i:i+params.patchSize, j:j+params.patchSize]
                error = alpha * np.sum(np.sqrt(np.sum((textureROI - targetROI)**2, -1)))
                error += (1-alpha)*OverlapErrorVertical2( imgTarget, imgTexture, (px[0], px[1] - params.overlapWidth), (i, j - params.overlapWidth), params)
                errorArray[i, j] = error
        minError = np.min(errorArray)
        tolerance = minError * (1.1)
        for i in range(sample_height - params.patchSize):
            for j in range(params.overlapWidth, sample_width - params.patchSize ):
                if errorArray[i, j] <  tolerance:
                    PixelbestPatchList.append((i, j))
    #check for leftmost layer
    elif px[1] == 0:
        errorArray = (1e+9)*np.ones((sample_height, sample_width))
        for i in range(params.overlapWidth, sample_height - params.patchSize ):
            for j in range(sample_width - params.patchSize):
                textureROI=imgTexture[i:i+params.patchSize, j:j+params.patchSize]
                error = alpha * np.sum(np.sqrt(np.sum((textureROI - targetROI)**2, -1))) 
                error += (1-alpha)*OverlapErrorHorizntl2( imgTarget, imgTexture, (px[0] - params.overlapWidth, px[1]), (i - params.overlapWidth, j), params)
                errorArray[i, j] = error
        minError = np.min(errorArray)
        tolerance = minError * (1.1)
        for i in range(params.overlapWidth, sample_height - params.patchSize ):
            for j in range(sample_width - params.patchSize):
                if errorArray[i, j] <  tolerance:
                    PixelbestPatchList.append((i, j))
    #for pixel placed inside
    else:
        errorArray = (1e+9)*np.ones((sample_height, sample_width))
        for i in range(params.overlapWidth, sample_height - params.patchSize):
            for j in range(params.overlapWidth, sample_width - params.patchSize):
                textureROI=imgTexture[i:i+params.patchSize, j:j+params.patchSize]
                error = alpha * np.sum(np.sqrt(np.sum((textureROI - targetROI)**2, -1))) 
                error += (1-alpha)*OverlapErrorVertical2( imgTarget, imgTexture, (px[0], px[1] - params.overlapWidth), (i,j - params.overlapWidth), params)
                error += (1-alpha)*OverlapErrorHorizntl2( imgTarget, imgTexture, (px[0] - params.overlapWidth, px[1]), (i - params.overlapWidth,j), params)
                errorArray[i, j] = error
        minError = np.min(errorArray)
        tolerance = minError * (1.1)
        for i in range(params.overlapWidth, sample_height - params.patchSize ):
            for j in range(sample_width - params.patchSize):
                if errorArray[i, j] <  tolerance:
                    PixelbestPatchList.append((i, j))
        
    return PixelbestPatchList

def chooseBestInitialPatch(imgSrc, imgTexture, params):
    patchSize = params.patchSize
    srcROI = imgSrc[0:patchSize, 0:patchSize]

    diffarray = cv2.matchTemplate(imgTexture, srcROI, cv2.TM_SQDIFF)
    print("diffarray.shape", diffarray.shape)
    
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(diffarray)
    patch = imgTexture[minLoc[1]:minLoc[1]+patchSize, minLoc[0]:minLoc[0]+patchSize]
    return patch

def transfer(imgSrc, imgTexture, params):
    print(imgSrc.shape)
    uheight, uwidth = params.userDesiredSize
    img_height = int((uheight // params.patchSize)*params.patchSize+params.patchSize)
    img_width = int((uwidth// params.patchSize)*params.patchSize+params.patchSize)
    outputImg = np.zeros((img_height, img_width, 3), np.uint8)
    print(outputImg.shape)
    outputImg[0:uheight, 0:uwidth] = imgSrc.copy()

    #Picking random patch to begin
    initialPatch = chooseBestInitialPatch(imgSrc, imgTexture, params)
    outputImg[0:params.patchSize,0:params.patchSize] = initialPatch

    #initializating next 
    growPatchLocation = (0,params.patchSize)
    drawDumpImg(outputImg, imgTexture, (0,0), list(), 0, params.patchSize, params.overlapWidth, 0)

    pixelsCompleted = 0
    totalPatches = ( (img_height)/ params.patchSize )*((img_width)/ params.patchSize) - 1
    sys.stdout.write("Progress : [%-20s] %d%% | PixelsCompleted: %d | thresholdConstant: --.------" % ('='*(int)(pixelsCompleted*20/totalPatches), (100*pixelsCompleted)/totalPatches, pixelsCompleted))
    sys.stdout.flush()
    while pixelsCompleted<totalPatches:
        thresholdConstant = params.initialThresConstant
        #set progress to zer0
        progress = 0 
        while progress == 0:
            #Get Best matches for current pixel
            bestPatchList = GetBestPatches(growPatchLocation, outputImg, imgTexture, thresholdConstant, params)
            if len(bestPatchList) > 0:
                progress = 1
                #Make A random selection from best fit pxls
                sampleMatch = bestPatchList[ randint(0, len(bestPatchList) - 1) ]
                drawDumpImg(outputImg, imgTexture, growPatchLocation, bestPatchList, sampleMatch, params.patchSize, params.overlapWidth, pixelsCompleted+1)
                FillImage( outputImg, imgTexture, growPatchLocation, sampleMatch, params)
                #Quilt this with in curr location
                QuiltPatches( outputImg, imgTexture, growPatchLocation, sampleMatch, params)
                #upadate cur pixel location
                growPatchLocation = (growPatchLocation[0], growPatchLocation[1] + params.patchSize)
                if growPatchLocation[1] + params.patchSize > img_width:
                    growPatchLocation = (growPatchLocation[0] + params.patchSize, 0)
            #if not progressed, increse threshold
            else:
                thresholdConstant *= 1.1
        pixelsCompleted += 1
        # print pixelsCompleted, thresholdConstant
        sys.stdout.write('\r')
        sys.stdout.write("Progress : [%-20s] %d%% | PixelsCompleted: %d | thresholdConstant: %f" % ('='*(int)(pixelsCompleted*20/totalPatches), (100*pixelsCompleted)/totalPatches, pixelsCompleted, thresholdConstant))
        sys.stdout.flush()

    outputImg=outputImg[0:uheight,0:uwidth]
    return outputImg

#Image Loading and initializations
# srcImgName = str(sys.argv[1])
# textureImgName = str(sys.argv[2])

# gPatchSize = int(sys.argv[3])
# gOverlapWidth = int(sys.argv[4])
# gInitialThresConstant = float(sys.argv[5])

srcImgName="bill.jpg"
textureImgName="rice.jpg"
gPatchSize = 10
gOverlapWidth=3
gInitialThresConstant = 30.0
alpha = 0.6

imgSrc = cv2.imread(srcImgName)
imgTexture = cv2.imread(textureImgName)
gUserDesiredSize = imgSrc.shape[0:2]
params = Params(gPatchSize, gOverlapWidth, gInitialThresConstant, gUserDesiredSize)

img = transfer(imgSrc, imgTexture, params)
makeGif("./dump/", "./dump/dump.gif", 1)

# Displaying Images
cv2.imshow('Sample Texture',imgTexture)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow('Generated Image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
