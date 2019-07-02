#!/usr/bin/env python

from common import *

#---------------------------------------------------------------------------------------#
#|                      Best Fit Patch and related functions                           |#
#---------------------------------------------------------------------------------------#
def GetBestPatches( px , imgTarget, imgSample, thresholdOverlapError, params):#Will get called in GrowImage
    PixelList = []
    sample_height, sample_width = imgSample.shape[0:2]

    #check for top layer
    if px[0] == 0:
        for i in range(sample_height - params.patchSize):
            for j in range(params.overlapWidth, sample_width - params.patchSize ):
                error = OverlapErrorVertical2( imgTarget, imgSample, (px[0], px[1] - params.overlapWidth), (i, j - params.overlapWidth), params)
                if error  < thresholdOverlapError:
                    PixelList.append((i,j))
                elif error < thresholdOverlapError/2:
                    return [(i,j)]
    #check for leftmost layer
    elif px[1] == 0:
        for i in range(params.overlapWidth, sample_height - params.patchSize ):
            for j in range(sample_width - params.patchSize):
                error = OverlapErrorHorizntl2( imgTarget, imgSample, (px[0] - params.overlapWidth, px[1]), (i - params.overlapWidth, j), params)
                if error  < thresholdOverlapError:
                    PixelList.append((i,j))
                elif error < thresholdOverlapError/2:
                    return [(i,j)]
    #for pixel placed inside
    else:
        for i in range(params.overlapWidth, sample_height - params.patchSize):
            for j in range(params.overlapWidth, sample_width - params.patchSize):
                error_Vertical   = OverlapErrorVertical2( imgTarget, imgSample, (px[0], px[1] - params.overlapWidth), (i,j - params.overlapWidth), params)
                error_Horizntl   = OverlapErrorHorizntl2( imgTarget, imgSample, (px[0] - params.overlapWidth, px[1]), (i - params.overlapWidth,j), params)
                if error_Vertical  < thresholdOverlapError and error_Horizntl < thresholdOverlapError:
                    PixelList.append((i,j))
                elif error_Vertical < thresholdOverlapError/2 and error_Horizntl < thresholdOverlapError/2:
                    return [(i,j)]
    return PixelList

def synthesis(imgSample, params):
    uheight, uwidth = params.userDesiredSize
    img_height = int((uheight // params.patchSize)*params.patchSize+params.patchSize)
    img_width = int((uwidth// params.patchSize)*params.patchSize+params.patchSize)
    sample_width = imgSample.shape[1]
    sample_height = imgSample.shape[0]
    img = np.zeros((img_height,img_width,3), np.uint8)

    #Picking random patch to begin
    randomPatchHeight = randint(0, sample_height - params.patchSize)
    randomPatchWidth = randint(0, sample_width - params.patchSize)
    img[0:params.patchSize,0:params.patchSize] = imgSample[randomPatchHeight:randomPatchHeight+params.patchSize, randomPatchWidth:randomPatchWidth+params.patchSize]
    #initializating next 
    GrowPatchLocation = (0,params.patchSize)
    drawDumpImg(img, imgSample, (0,0), list(), 0, params.patchSize, params.overlapWidth, 0)

    pixelsCompleted = 0
    TotalPatches = ( (img_height)/ params.patchSize )*((img_width)/ params.patchSize) - 1
    sys.stdout.write("Progress : [%-20s] %d%% | PixelsCompleted: %d | ThresholdConstant: --.------" % ('='*(int)(pixelsCompleted*20/TotalPatches), (100*pixelsCompleted)/TotalPatches, pixelsCompleted))
    sys.stdout.flush()
    while pixelsCompleted<TotalPatches:
        ThresholdConstant = params.initialThresConstant
        #set progress to zer0
        progress = 0 
        while progress == 0:
            thresholdOverlapError = ThresholdConstant * params.patchSize * params.overlapWidth
            #Get Best matches for current pixel
            List = GetBestPatches(GrowPatchLocation, img, imgSample, thresholdOverlapError, params)
            if len(List) > 0:
                progress = 1
                #Make A random selection from best fit pxls
                sampleMatch = List[ randint(0, len(List) - 1) ]
                drawDumpImg(img, imgSample, GrowPatchLocation, List, sampleMatch, params.patchSize, params.overlapWidth, pixelsCompleted+1)
                FillImage( img, imgSample, GrowPatchLocation, sampleMatch, params)
                #Quilt this with in curr location
                QuiltPatches( img, imgSample, GrowPatchLocation, sampleMatch, params)
                #upadate cur pixel location
                GrowPatchLocation = (GrowPatchLocation[0], GrowPatchLocation[1] + params.patchSize)
                if GrowPatchLocation[1] + params.patchSize > img_width:
                    GrowPatchLocation = (GrowPatchLocation[0] + params.patchSize, 0)
            #if not progressed, increse threshold
            else:
                ThresholdConstant *= 1.1
        pixelsCompleted += 1
        # print pixelsCompleted, ThresholdConstant
        sys.stdout.write('\r')
        sys.stdout.write("Progress : [%-20s] %d%% | PixelsCompleted: %d | ThresholdConstant: %f" % ('='*(int)(pixelsCompleted*20/TotalPatches), (100*pixelsCompleted)/TotalPatches, pixelsCompleted, ThresholdConstant))
        sys.stdout.flush()

    img=img[0:uheight,0:uwidth]
    return img

#Image Loading and initializations
inputName = str(sys.argv[1])
gPatchSize = int(sys.argv[2])
gOverlapWidth = int(sys.argv[3])
gInitialThresConstant = float(sys.argv[4])
gUserDesiredSize = (250, 250)

params = Params(gPatchSize, gOverlapWidth, gInitialThresConstant, gUserDesiredSize)

imgSample = cv2.imread(inputName)

img = synthesis(imgSample, params)
makeGif("./dump/", "./dump/dump.gif", 1)

# Displaying Images
cv2.imshow('Sample Texture',imgSample)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow('Generated Image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
