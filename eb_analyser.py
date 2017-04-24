# -*- coding: utf-8 -*-
# import the packages we need
from PIL import Image, ImageDraw
import numpy as np
import sys
import math
import os
import glob
import time
import winsound
import win32com.client 

## USER VARIABLES ##############################################################

# OUTPUT FOLDER, CSV FILE, AND ERROR FILE
outputDir = "C:\Users\Hanne Stawarz\Documents\EBOutput\\" # NOTE don't forget to end with \\
csvFile = "C:\Users\Hanne Stawarz\Documents\EBOutput\output.csv"
errFile = "C:\Users\Hanne Stawarz\Documents\EBOutput\err.txt"
testDir = "C:\Users\Hanne Stawarz\Documents\\"

## EB RECOGNITION PARAMETERS ##

# extra boundary around bounding box, as a scale factor (e.g. 1.5 = 50% larger, 2 = 100% larger)
boxMargin = 2

# defines minimum box size - EBs smaller than this no of pixels not considered
minBoxWidth = 4
maxBoxWidth = 16

# Currently considers only pixels exceeding this brightness value (where 0 = black and 255 = white)
threshold = 200

# Threshold brightness for detecting EBs in binary image (e.g. pixels brighter than this will be set to 255, otherwise 0)
detectionThresholdVal = 80 # Used to be 30, then 120, then 80

# Threshold determining how bright an EB stain should be to be considered viable
aveBrightnessThreshold = 20 # used to be 20

# Minimum brightness for a stack to have to be considered viable, in range 0 - 255
minStackBrightness = 10 # used to be 25

failedStainCount = 0

minCircularity = 0.5

## POLARITY SCORING PARAMETERS ##

# Polarity Score Interval - given as proportion (e.g. 0.1 is 10%) of CoB difference
ps0 = 0.06 #0.12
ps1 = 0.1 #0.24
ps2 = 0.16 #0.36

## CONFIDENCE CALCULATION PARAMETERS ##

# Max and min brightness levels to interpolate between
brightnessUpper = 60
brightnessLower = 20

## CONFIDENCE WEIGHTINGS
# 5 weights. thus equal weighting would be 1/5
# NOTE: ALL WEIGHTINGS SHOULD ADD TO 1

numConsideredWeight = 0.1
brightnessWeight    = 0.225
sizeWeight          = 0.25
stainWeight         = 0.25
circularityWeight   = 0.175


## FUNCTIONS ###################################################################

def getThresholdByPercentage(proportion, imArr):
    cumSum = 0.0
    for x in range(0, len(imArr)):
        for y in range(0, len(imArr[0])):
            cumSum += imArr[x, y]
    return int(cumSum * proportion)

## makes brightness value range from 0-255 ('re-scales')
def equalise(imArr):

    width = len(imArr[0])
    height = len(imArr)

    lowest = np.amin(imArr)
    temp = imArr.copy()
    temp -= lowest
    highest = np.amax(temp)
    if highest < minStackBrightness:
        print "Below minstack brightness (%d)" % lowest
        temp = None
    else:
        r = 255.0 / highest
        for x in range(0, width):
            for y in range(0, height):
                equalisedValue = temp[y,x] * r
                temp[y,x] = int(equalisedValue)
    return temp

def findCenter(imArr, width, height, threshold):
    #initialises array 'weight' as zero, ratio as zero, and perceived centre of x and y co-ordinates as zero
    weight = 0
    r = 0.0
    cX = 0.0
    cY = 0.0
    #iterates through every pixel in the cropped EB image, considers only pixels above threshold
    for x in range(0, height):
        for y in range(0, width):
            if imArr[x, y] < threshold:
                continue
            # calculates ratio, allows it to be a non-integer
            # divides brightness at given co-ordinates by brightness " plus total array weight
            r = float(imArr[x,y]) / (weight + imArr[x,y])
            # ratio 'weights' the contribution of brightness per co-ordinate, this is added to each centre of brightness co-ordinate cX and cY
            cX += (r * (y - cX))
            cY += (r * (x - cY))
            # brightness at each co-ordinate is added to the 'weight' of pixels considered thus far - thus each new pixel becomes less influential
            weight += imArr[x,y]
    return cX, cY

def getAverageValue(imArr, width, height):
    cumSum = 0
    for x in range(0, height):
        for y in range(0, width):
            cumSum += imArr[x, y]

    cumSum /= (width * height)
    return cumSum

def generateSortedArray(imArr, width, height):
    # First, generate an array of the brightness values
    # Should be stored in tuple as follows:
    # (xCoord, yCoord, brightness)
    tempArray = []
    for x in range(0, height):
        for y in range(0, width):
            tempTup = (x, y, imArr[x,y])
            tempArray.append(tempTup)

    # Then sort it
    # NOTE: May be much faster to sort it as we build it
    tempArray.sort(key=lambda tup: tup[2])
    tempArray.reverse()

    return tempArray

def findCenterByDescentPercentage(arr, threshold):
    weight = 0
    r = 0.0
    cX = 0.0
    cY = 0.0
    count = 0;
    cumSum = 0;
    for count in range(0, len(arr)):
        cumSum += arr[count][2]
        if cumSum > threshold and arr[count][2] < 250:
            break

        r = float(arr[count][2]) / (weight + arr[count][2])
        # TODO fix this confusion; index by y first, then x
        tempdX = (r * (arr[count][1] - cX))
        tempdY = (r * (arr[count][0] - cY))

        cX += tempdX
        cY += tempdY
        weight += arr[count][2]
    return cX, cY, count

def findCenterByDescent(arr, threshold, brightness):
    weight = 0
    r = 0.0
    cX = 0.0
    cY = 0.0
    count = 0;
    for count in range(0, len(arr)):

        if arr[count][2] == 0 or arr[count][2] < brightness:
            break;
        r = float(arr[count][2]) / (weight + arr[count][2])
        # TODO fix this confusion; index by y first, then x
        tempdX = (r * (arr[count][1] - cX))
        tempdY = (r * (arr[count][0] - cY))
        motion = math.sqrt(tempdX ** 2 + tempdY ** 2)
        #print motion
        if(motion < threshold):
            print "Below threshold - breaking! Considered %d" % count
            break;
        else:
            cX += tempdX
            cY += tempdY
            weight += arr[count][2]
    return cX, cY, count

def eraseEB(img, minx, miny, maxx, maxy):
    for y in range(miny, maxy+1):
        for x in range(minx, maxx+1):
            img[y, x] = 0

    return img

def getCumulativeStack(imgs):
    # np.zeros 'initialises' the function with an empty array, size defined below
    cum = np.zeros((512, 512))

    #
    for i in range (0, len(imgs)):
        # iterates through images in stack (in order), opens and resizes, converts to array and adds all together (cumulative i.e. cum)
        im = Image.open(imgs[i])
        im.thumbnail((512, 512), Image.NEAREST)
        imArr = np.asarray(im)
        cum += imArr

    return cum

def getEqualisedStack(imgs):
    cum = getCumulativeStack(imgs)

    #
    cum = equalise(cum)
    #Image.fromarray(cum).show()
    return cum

def processImg(img):

    # converts to luminosity
    if convertFlag:
        img = img.convert('I')
        
    img = img.convert('L')

    # converts to array, defines as imArr and equalises using prior function
    imArr = np.asarray(img)


    imArr = equalise(imArr)

    # TODO better error handling needed
    if imArr is None:
        return None, None, None, None, None, None, None
        
    #### THRESHOLD: AVERAGE VALUE APPROACH
    #threshold = getAverageValue(imArr, img.size[0], img.size[1])

    # uses centre of brightness function to return centre of brightness co-ordinates at x and y
    #cobX, cobY = findCenter(imArr, img.size[0], img.size[1], threshold)

    #### THRESHOLD: DESCENT APPROACHH
    ## Perhaps a ratio is better.. brightness to distance moved....
    ## Maybe keep track of the last 5 movements and stop when range is too small

    # NOTE: The threshold should be given as a value between 0 and 1
    threshold = getThresholdByPercentage(0.1, imArr)

    sortedArray = generateSortedArray(imArr, img.size[0], img.size[1])

    cobX, cobY, numConsidered = findCenterByDescentPercentage(sortedArray, threshold)

    meanBrightness = 0
    numNonBlack = 0

    for y in range(0, img.size[1]):
        for x in range(0, img.size[0]):
            if imArr[y, x] > 0:
                numNonBlack += 1
                meanBrightness += imArr[y,x]
    
    if numNonBlack > 0:
        meanBrightness = meanBrightness / numNonBlack
    else:
        meanBrightness = 0

    brightnessRatio = numNonBlack / float(img.size[0] * img.size[1])

    print "BRIGHTNESS RATIO IS %.2f" % (numNonBlack / float(img.size[0] * img.size[1]))

    #### THRESHOLD: TOP 10% APPROACH
    # img = img.convert(L)
    # np.asarray(img) - convert to array
    # calculate array sum, multiply by 0.1 and store as ten percent value
    # generate sorted array as above
    # for i in range - len(img), add to cumulative
    # if 0.1xtotal array is reached, break
    # set result as threshold

    # uses pythagoras to find the distance between the centre of the image and centre of brightness (hypotenuse!)
    dist = math.sqrt(math.pow((cobX - (img.size[0] / 2)), 2) + math.pow((cobY - (img.size[0] / 2)), 2))

    # generates new RGB image which is the same size as the cropped square (prev was luminosity only i.e. BW) - uses prev array to create this
    rgbimg = Image.new("RGB", img.size)
    rgbimg.paste(Image.fromarray(imArr))

    # draws line from centre of image (divide image size, width and height, by 2) to centre of brightness
    draw = ImageDraw.Draw(rgbimg)
    draw.line([img.size[0] / 2, img.size[1] / 2, int(round(cobX)), int(round(cobY))],fill="red",width=1)
    draw.point([img.size[0]/2, img.size[1]/2], fill="#ffff00")
    draw.point([int(round(cobX)), int(round(cobY))], fill="#0000ff")
    # 'free up' draw object once line drawn
    del draw

    # NOTE: will be calculated in future; for now, dummy value of 0 (ignore)

    return rgbimg, cobX, cobY, dist, numConsidered, meanBrightness, brightnessRatio

def fillHoles(thresholded, minNeighbours):
    
    filled = thresholded.copy()
    
    for y in range(1,511):
        for x in range(1, 511):
            if thresholded[y, x] == 0:
                value = int(thresholded[y-1,x]) + int(thresholded[y,x-1]) + int(thresholded[y+1,x]) + int(thresholded[y,x+1])
                if (value / 255) >= minNeighbours:
                    filled[y,x] = 255
    return filled
            
    
def removeNoise(thresholded, minNeighbours):
    denoised = thresholded.copy()
    
    for y in range(1,511):
        for x in range(1, 511):
            if thresholded[y, x] == 255:
                value = int(thresholded[y-1,x]) + int(thresholded[y,x-1]) + int(thresholded[y+1,x]) + int(thresholded[y,x+1])
                if (value / 255) < minNeighbours:
                    denoised[y,x] = 0
    return denoised

def getCandidateEBs(eqMaxproj):

    # Created (new) temporary image 'object' from the 2D array equalised stack ('max projection') and defines as 'temp'. Converts temp to luminosity only* could remove
    # Converts...back to array (grey) and copies it
    temp = Image.fromarray(eqMaxproj)
    if convertFlag:
        temp = temp.convert('I')
    gray = temp.convert('L')

    thresholded = np.asarray(gray).copy()

    # Create 'blobs' by setting all pixels with brightness <detectionThreshold as 0 (black) and all >detectionThreshold as white (255)
    thresholded[thresholded < detectionThresholdVal] = 0
    thresholded[thresholded >= detectionThresholdVal] = 255

    ogThresholded = thresholded

    # Clean up thresholded image by filling small gaps and removing lone pixels
    # (Note the order of operations below have been defined arbitrarily)
    thresholded = fillHoles(thresholded, 2)
    thresholded = fillHoles(thresholded, 2)
    thresholded = removeNoise(thresholded,2)               
    thresholded = fillHoles(thresholded, 3)
    thresholded = removeNoise(thresholded,3)        
    
    #  Initialises empty array for ebs (later returned)
    ebs = []

    # Iterates across every pixel in (resized!) image - all 512 of them!
    for y in range (0, 512):
        for x in range (0, 512):
            # Ignores those set to zero (black) by threshold and continues reading
            if thresholded[y, x] == 0:
                continue
            elif y == 511:
                continue
            elif thresholded[y+1, x] == 0:
                continue
            else:

                # min and max co-ordinates define the boundaries of the pixels. yTemp and xTemp save temporary co-ordinates which iterate through each pixel
                # the above prevents x and y being changed before their neighbours are known
                minx = x
                maxx = x
                miny = y
                maxy = y

                yTemp = y
                xTemp = x
                
                rightFound = False
                leftFound = False
                bottomFound = False
                
                #  Starting from the top edge, expand the bounding box left, right, and down until it encompasses the EB
                while not (rightFound and leftFound and bottomFound):
                    
                    # Expand to the right
                    yTemp = miny
                    while yTemp <= maxy:

                        if maxx + 1 == 512:
                            rightFound = True
                            break
                            
                        if thresholded[yTemp, maxx + 1] == 255:
                            maxx += 1
                            bottomFound = False
                            yTemp = miny
                            continue
                        elif yTemp == maxy:
                            rightFound = True

                        yTemp += 1
                    
                    # Expand to the left
                    yTemp = miny
                    while yTemp <= maxy:

                        if minx == 0:
                            leftFound = True
                            break

                        if thresholded[yTemp, minx - 1] == 255:
                            minx -= 1
                            bottomFound = False
                            yTemp = miny
                            continue
                        elif yTemp == maxy:
                            leftFound = True
                            
                        yTemp += 1

                        
                    # Expand to the bottom
                    xTemp = minx
                    while xTemp <= maxx:

                        if maxy == 511:
                            bottomFound = True
                            break

                        if thresholded[maxy + 1, xTemp] == 255:
                            maxy += 1
                            leftFound = False
                            rightFound = False
                            xTemp = minx
                            continue
                        elif xTemp == maxx:
                            bottomFound = True
                            
                        xTemp += 1
                    

                # Ignore if it's an edge boi
                if maxx == 511 or maxy == 511 or minx == 0:
                    print "DISREGARDING: EB at edge of image, coordinates " + str(minx) + ", " + str(miny)
                    continue

                # Defines box size (width and height via x and y co-ordinates on either side). Radius via dividing by 2
                hDist = (maxx - minx + 1) / 2.0
                vDist = (maxy - miny + 1) / 2.0

                # Makes halfBoxWidth the largest dimension, whether that be horizontal or vertical
                halfBoxWidth = hDist if hDist > vDist else vDist

                estArea = (halfBoxWidth * halfBoxWidth) * math.pi

                # Get actual area of EB for comparison to check how spherical it is
                areaSum = 0
                for i in range(miny, maxy + 1):
                    for j in range(minx, maxx + 1):
                        if(thresholded[i][j] == 255):
                            areaSum += 1

                # Replace thresholded image with new image in which the EB is erased, to prevent the algorithm identifying a 'new' EB at each row (which is actually the same EB)
                thresholded = eraseEB(thresholded, minx, miny, maxx, maxy)

                # defined above as arbitrary minimum size of EB, below which identified spots are ignored/not boxed
                if halfBoxWidth < minBoxWidth:
                    print "DISREGARDING: undersized candidate EB at coordinates " + str(minx) + ", " + str(miny)
                    continue

                if halfBoxWidth >= maxBoxWidth:
                    print "DISREGARDING: oversized candidate EB"
                    continue

                # Check difference of area to expected area to assess circularity
                # If perfect match, score is 1
                print "maxx: %d; minx: %d; maxy: %d; miny: %d" % (maxx, minx, maxy, miny)
                print "radius: %d; hDist: %.1f; vDist: %.1f; estArea: %.2f, areaSum: %d" % (halfBoxWidth, hDist, vDist, estArea, areaSum)
                circularity = 1.0 - math.fabs((float(areaSum) / float(estArea)) - 1.0)
                if circularity < minCircularity:
                    print "DISREGARDING: EB at " + str(minx) + ", " + str(miny) + " non-circular with score %f (may be overlap)" % circularity
                    continue

                # Store diameter before scaling box by margin
                ebDiam = halfBoxWidth * 4
                
                if (ebDiam * 16) > 1000:
                    print "DISREGARDING: suspected RB of size %dnm" % (ebDiam * 16)
                    continue

                # Increases boxwidth by our defined safety margin (defined above as 1.5 i.e. 50% larger than native box size)
                halfBoxWidth *= boxMargin

                # Finds centre of each co-ordinate
                centerX = (minx + maxx) / 2
                centerY = (miny + maxy) / 2

                # Subtract centre from the boxwidth to find origin for box margin
                xOrigin = centerX - halfBoxWidth
                yOrigin = centerY - halfBoxWidth

                # TODO do this more neatly.. avoids crashing on negative index
                if xOrigin < 0 or yOrigin < 0:
                    print "DISREGARDING: Edge boy"
                    continue

                # Multiply by two to get the full box width
                boxWidth = halfBoxWidth * 2

                # Multiply all coords by 2 to translate to full-size (1024x1024) coords
                # Note that the EB analysis performed on full 1024x1024 img
                # Identification performed on 512x512 in interest of speed
                boxWidth *= 2
                xOrigin *= 2
                yOrigin *= 2

                # Initialises empty 'dictionary'
                thisEB = {}
                # Defines EB 'id' (number) as order in which it is identified - allows unique naming
                thisEB['id'] = len(ebs)
                # int de-floats. Defines values in dictionary (defines object properties) such as xmin as those identified above, can be used to draw box, identify centre
                thisEB['xMin'] = int(xOrigin)
                thisEB['yMin'] = int(yOrigin)
                thisEB['xMax'] = min(int(xOrigin + boxWidth), 1023)
                thisEB['yMax'] = min(int(yOrigin + boxWidth), 1023)
                thisEB['pixelSize'] = ebDiam
                thisEB['pixelArea'] = areaSum
                thisEB['circularity'] = circularity
                thisEB['size'] = ebDiam * 16

                # appends EB dictionary with pertinent information/properties to the list 'ebs' which is returned below, can thus be analysed
                ebs.append(thisEB)
                print "Found an EB at coordinates " + str(int(xOrigin)) + ", " + str(int(yOrigin))

    outImg = Image.fromarray(ogThresholded)
    rgbimg = Image.new("RGB", outImg.size)
    rgbimg.paste(outImg)

    draw = ImageDraw.Draw(rgbimg)
    for eb in ebs:
        draw.line((eb['xMin'],eb['yMin'],eb['xMax'],eb['yMin']),fill="red",width=1)
        draw.line((eb['xMax'],eb['yMin'],eb['xMax'],eb['yMax']),fill="red",width=1)
        draw.line((eb['xMax'],eb['yMax'],eb['xMin'],eb['yMax']),fill="red",width=1)
        draw.line((eb['xMin'],eb['yMax'],eb['xMin'],eb['yMin']),fill="red",width=1)
    del draw

    return ebs

### Find index of brightest slice for each stain
def findEachBrightestSlice(genSlices, polSlices, ebs):
    for eb in ebs:

        # curMaxBrightness keeps track of the greatest 'brightness' value seen in a slice for each channel
        curMaxBrightnessGen = 0
        curMaxBrightnessPol = 0

        # Saves the index of the brightest slice
        # Initialised to -1 just to identify a failed run; will be overwritten immediately. Not initialised as zero (zero may falsely appear as valid value)
        genBrightestSlice = -1
        polBrightestSlice = -1
        
        dim = 1024
        
        genProj = np.zeros((dim, dim))
        genProjCount = 0
        polProj = np.zeros((dim, dim))
        polProjCount = 0

        # iterates through each slice, length defined by number of z_00 etc
        for i in range (0, len(genSlices)):

            # Initialise sum of slice brightness to zero (general stain)
            curGenSum = 0;
            # Opens the general slice - [i] defines iteration
            curGenSlice = Image.open(genSlices[i])
            # Converts to luminosity
            if convertFlag:
                curGenSlice = curGenSlice.convert('I')
            curGenSlice = curGenSlice.convert('L')
            # Converts to array so calculations can be performed (array per slice)
            curGenSlice = np.asarray(curGenSlice)

            # same as above but in polar candidate channel
            curPolSum = 0;
            curPolSlice = Image.open(polSlices[i])
            if convertFlag:
                curPolSlice = curPolSlice.convert('I')
            curPolSlice = curPolSlice.convert('L')
            curPolSlice = np.asarray(curPolSlice)

            # Looks within box range (co-ordinates defined previously), adds pixel brightness to total box brightness per slice                        
            for x in range (eb['xMin'], eb['xMax']):
                for y in range (eb['yMin'], eb['yMax']):
                    curGenSum += curGenSlice[y, x]
                    curPolSum += curPolSlice[y, x]

            # Add to cumulative slicebit thing
            if curGenSum > 10:
                genProjCount += 1
                genProj += curGenSlice
            if curPolSum > 10:
                polProjCount += 1
                polProj += curPolSlice

            # curGenSum is sum of pixel values within EB box (gen stain) - if a higher brightness is found, this now becomes curMaxBrightnessGen
            # brightest slice is also set to 'i'
            if curGenSum > curMaxBrightnessGen:
                curMaxBrightnessGen = curGenSum
                genBrightestSlice = i

            # as above but for polar candidate channel
            if curPolSum > curMaxBrightnessPol:
                curMaxBrightnessPol = curPolSum
                polBrightestSlice = i
        
        #genProj /= float(genProjCount)
        #polProj /= float(polProjCount)
        
        #genProj = equalise(genProj)
        #polProj = equalise(polProj)
        
        eb['genProj'] = genProj
        eb['polProj'] = polProj
    
        # TODO trialling new method, leaving commented out for now...
        """
        print "Getting brightness values..."
        
        # Get the mean brightness for the best slices                    
        meanBrightnessPol = 0
        meanBrightnessGen = 0
        numNonBlackPol = 0
        numNonBlackGen = 0
        
        curGenSlice = Image.open(genSlices[genBrightestSlice])
        curGenSlice = curGenSlice.convert('L')
        curGenSlice = np.asarray(curGenSlice)
        curGenSlice = equalise(curGenSlice)
        
        curPolSlice = Image.open(polSlices[polBrightestSlice])
        curPolSlice = curPolSlice.convert('L')
        curPolSlice = np.asarray(curPolSlice)
        curPolSlice = equalise(curPolSlice)
        
        for y in range(eb['yMin'], eb['yMax']):
            for x in range (eb['xMin'], eb['xMax']):
                if curGenSlice[y, x] > 0:
                    numNonBlackGen += 1
                    meanBrightnessGen += curGenSlice[y,x]
                if curPolSlice[y,x] > 0:
                    numNonBlackPol += 1
                    meanBrightnessPol += curPolSlice[y,x]
        
        if numNonBlackGen > 0:
            meanBrightnessGen /= numNonBlackGen
        if numNonBlackPol > 0:
            meanBrightnessPol /= numNonBlackPol
        """

        # Creates new property in dictionary (the brightest slice index in a stack e.g. z_10) assigned from analysis above. For each stain
        eb['genMaxSlice'] = genBrightestSlice

        eb['polMaxSlice'] = polBrightestSlice


    return ebs

### This function finds the centre of brightness for each stain, as well as the distance and angle between the two
def processEBs(genChannel, polChannel, ebs):

    for eb in ebs:

        # Opens brightest slice in the general channel, indexed in genMaxSlice dictionary property
        gChanSlice = Image.open(genChannel[eb['genMaxSlice']])

        
        # Crops the brightest slice to the constraints set by the max/min co-ordinates (includes 'safety margin' of 1.5)
        gChanImg = gChanSlice.crop((eb['xMin'],eb['yMin'],eb['xMax'],eb['yMax']))
        # Saves cropped image as property of EB dictionary (unique to each EB)

        eb['genChanImg'] = gChanImg.convert(mode="RGB")
        
        if convertFlag:
            gChanImg = gChanImg.convert('I')

        # as above but for polar candidate channel
        rChanSlice = Image.open(polChannel[eb['polMaxSlice']])

        
        
        rChanImg = rChanSlice.crop((eb['xMin'],eb['yMin'],eb['xMax'],eb['yMax']))
        eb['polChanImg'] = rChanImg.convert(mode="RGB")
        if convertFlag:
            rChanImg = rChanImg.convert('I')

        eb['genAveBrightness'] = getAverageValue(np.asarray(gChanImg), gChanImg.size[0], gChanImg.size[1])

        # Adds processImg function output as dictionary definitions to dictionary 'eb', includes:
        #genChanImgProcessed is final processed general channel image
        #genCobX and Y are X and Y co-ordinates of the centre of brightness
        #genCoBDist is the distance between the centre of the cropped EB and the centre of brightness
        #genNumConsidered outputs how many pixels are considered when calculating the CoB, at the moment threshold is arbitrary but later this will be more informative (sorted threshold by descent)
        
        eb['genChanImgProcessed'], eb['genCoBX'], eb['genCoBY'], eb['genCoBDist'], eb['genNumConsidered'], eb['genBrightness'], eb['genBrightnessRatio'] = processImg(gChanImg)

        # TODO Need better error handling
        if eb['genChanImgProcessed'] is None:
            print "Failed on general channel processing of eb " + str(eb['id']) + "; too dark"
            eb['failed'] = True
            continue


        eb['polAveBrightness'] = getAverageValue(np.asarray(rChanImg), rChanImg.size[0], rChanImg.size[1])
        eb['polChanImgProcessed'], eb['polCoBX'], eb['polCoBY'], eb['polCoBDist'], eb['polNumConsidered'], eb['polBrightness'], eb['polBrightnessRatio'] = processImg(rChanImg)

        if eb['polChanImgProcessed'] is None:
            print "Failed on candidate polar channel processing of eb " + str(eb['id']) + "; too dark"
            eb['failed'] = True
            continue

        # finds centre of image (not CoB)
        eb['xCenter'] = gChanImg.size[0] / 2
        eb['yCenter'] = gChanImg.size[1] / 2

        # v is vector. Finds vectors of distance between CoB and CoI (in x and y)
        # Normalised to magnitude of 1 so angle can later be calculated
        vGen = np.array([ (eb['genCoBX'] - eb['xCenter']), (eb['genCoBY'] - eb['yCenter']) ])
        vGenNorm = np.linalg.norm(vGen)
        vGen[0] = vGen[0] / vGenNorm
        vGen[1] = vGen[1] / vGenNorm

        # same for polar candidate channel
        vPol = np.array([ (eb['polCoBX'] - eb['xCenter']), (eb['polCoBY'] - eb['yCenter']) ])
        vPolNorm = np.linalg.norm(vPol)
        vPol[0] = vPol[0] / vPolNorm
        vPol[1] = vPol[1] / vPolNorm

        # uses cosine rule to find angle between two vectors above, saved as 'angleDif'
        dot = np.inner(vGen, vPol)
        angle = math.acos(dot)
        eb['angleDif'] = angle

        # Figure out direction of polarity
        xP = eb['polCoBX']
        yP = eb['polCoBY']
        xG = eb['genCoBX']
        yG = eb['genCoBY']

        dX = xP - xG;
        dY = yP - yG;

        polDir = 0;

        if dX < 0:
            if dY is not 0:
                polDir = 270 + ((math.atan(dY / dX) / math.pi) * 180)
            else:
                polDir = 270
        elif dX > 0:
            if dY is not 0:
                polDir = 90 + ((math.atan(dY / dX) / math.pi) * 180)
            else:
                polDir = 90
        else:
            if dY < 0:
                polDir = 0
            elif dY > 0:
                polDir = 180

        eb['polarityDir'] = polDir

        eb['failed'] = False

    return ebs


###

def getBrightnessScore(brightness):
    if brightness < brightnessLower:
        return 0.0
    elif brightness > brightnessUpper:
        return 1.0
    else:
        return (brightness - brightnessLower) / float(brightnessUpper - brightnessLower)

###
def generateOutput(filename, saveDir, ebs, stainQuality):
    for eb in ebs:

        if eb['failed'] is True:
            print "Could not process EB " + str(eb['id'])
            eb['genChanImg'].save(saveDir + "eb-" + str(eb['id']) + "-general.png", 'PNG')
            eb['polChanImg'].save(saveDir + "eb-" + str(eb['id']) + "-polarised.png", 'PNG')

            # opens analysis directory to save analysis text file per EB. A allows us to append to the file
            f = open(saveDir + 'the-hell-file.txt', 'a')
            f.write("//// EB " + str(eb['id']) + " Failed to process /////////////////////////\n\n")
            f.close()

            continue


        # saveDir is the directory redirected by pipistrelle, within which an analysis folder is created (stuff saved here)
        # cropped EB is saved in this directory named eb-[index from ID e.g. 0, 1, 2), -general and png (image type)
        # processed also saved - this cropped EB has its centre of brightness/CoI denoted by the line
        eb['genChanImg'].save(saveDir + "eb-" + str(eb['id']) + "-general.png", 'PNG')
        eb['genChanImgProcessed'].save(saveDir + "eb-" + str(eb['id']) + "-general-output.png", 'PNG')
        eb['polChanImg'].save(saveDir + "eb-" + str(eb['id']) + "-polarised.png", 'PNG')
        eb['polChanImgProcessed'].save(saveDir + "eb-" + str(eb['id']) + "-polarised-output.png", 'PNG')

        # opens analysis directory to save analysis text file per EB. A allows us to append to the file
        f = open(saveDir + 'eb-' + str(eb['id']) + '-analysis.txt', 'a')

        #
        f.write("//// RUN EXECUTED AT " + time.strftime("%c") + " /////////////////////////\n")
        f.write("// Using:\n")
        f.write("//  - CoB threshold = " + str(threshold) + "\n")
        f.write("//  - Average brightness threshold = " + str(aveBrightnessThreshold) + "\n")
        f.write("//  - boxMargin = " + str(boxMargin) + "\n")
        f.write("//  - minBoxWIdth = " + str(minBoxWidth) + "\n\n")

        f.write("  General Stain:\n")
        f.write("   - CoB Dist: " + str(eb['genCoBDist']) + "\n")
        f.write("   - numConsidered: " + str(eb['genNumConsidered']) + "\n")
        f.write("   - maxSlice: " + str(eb['genMaxSlice']) + "\n")
        f.write("   - Approx. size: " + str(eb['size']) + "nm\n")
        f.write("   - aveBrightness: " + str(eb['genAveBrightness']))
        if eb['genAveBrightness'] < aveBrightnessThreshold:
            f.write(" - BELOW THRESHOLD, POTENTIALLY NON-VIABLE")
        f.write("\n")

        f.write("\n  Candidate Polarised Stain:\n")
        f.write("   - CoB Dist: " + str(eb['polCoBDist']) + "\n")
        f.write("   - numConsidered: " + str(eb['polNumConsidered']) + "\n")
        f.write("   - maxSlice: " + str(eb['polMaxSlice']) + "\n")
        f.write("   - aveBrightness: " + str(eb['polAveBrightness']))
        if eb['polAveBrightness'] < aveBrightnessThreshold:
            f.write(" - BELOW THRESHOLD, POTENTIALLY NON-VIABLE")
        f.write("\n")

        f.write("\n  ----------------\n\n")
        angleDif = eb['angleDif'] * (180.0 / math.pi)
        f.write("  Angle difference: " + str(angleDif) + " degrees\n")

        # Get x distance and y distance between genCoB and polCoB
        xDist = eb['genCoBX'] - eb['polCoBX']
        yDist = eb['genCoBY'] - eb['polCoBY']
        
        print "xDist: %d, yDist: %d" % (xDist, yDist)
        
        xDistCenter = (eb['genChanImg'].size[0]/2.0) - eb['polCoBX']
        yDistCenter = (eb['genChanImg'].size[0]/2.0) - eb['polCoBY']
        
        print "xDistCenter: %d, yDistCenter: %d" % (xDistCenter, yDistCenter)

        # Use pythag to calc distance between CoBs, and check that it's positive (force it to be positive)
        pixelDistCoB = math.sqrt(math.pow(xDist, 2) + math.pow(yDist, 2))
        pixelDistCenter = math.sqrt(math.pow(xDistCenter, 2) + math.pow(yDistCenter, 2))
        
        pixelDistWeight = 1.0
        
        # Take the largest of the dist to the other CoB or the dist to the center
        # Approximate fix for some issues but NEEDS REFINEMENT
        if pixelDistCoB < pixelDistCenter:
            print "pixeldistcenter better: %d" % pixelDistCenter
        else:
            pixelDistWeight = 1.5 - (0.5 * (pixelDistCenter/pixelDistCoB))
            print "pixeldist better: %d, weight %.2f" % (pixelDistCoB, pixelDistWeight)
            
        pixelDist = pixelDistCoB if pixelDistCoB > pixelDistCenter else pixelDistCenter
        

        # Get pixel dist as a percentage of EB diameter
        percentDist = (pixelDist / float(eb['pixelSize'])) * 100


        # Write CoB to file and percentage toooo
        f.write("  CoB Difference:   " + str(pixelDist) + " pixels (%.1f%% of eb width)\n\n" % percentDist)
        f.write("////////////////////////////////////////////////////////////////\n\n")
        f.close()

        # sliceDif (note, unused for now)
        sliceDif = int(math.fabs(eb['genMaxSlice'] - eb['polMaxSlice']))


        # Get brightness scores for both stains and calculate brightnessfactor
        polBrightnessScore = getBrightnessScore(eb['polBrightness'])
        genBrightnessScore = getBrightnessScore(eb['genBrightness'])
                
        brightnessFactor = (genBrightnessScore + polBrightnessScore) / 2.0

        # Apply brightness ratio to stainQuality        
        if eb['genBrightness'] > eb['polBrightness']:
            stainQuality *= (eb['polBrightness'] / float(eb['genBrightness']))
        else:
            stainQuality *= (eb['genBrightness'] / float(eb['polBrightness']))


        #print "brightness pol: %d, gen %d" % (eb['polBrightness'], eb['genBrightness'])


        # Circularity Factor - anything above 0.5 is an automatic 1
        # See notes on circularity in GetCandidateEBs
        circularityFactor = (eb['circularity'] - 0.5) * 2

        # Size Factor
        # NOTE: brightness of gen channel may say something about size, TODO
        sizeFactor = 0
        
        # decrease linearly between 500 and 1000
        if eb['size'] > 200 and eb['size'] < 1000:
            if eb['size'] > 500:
                print "size: %d" % eb['size']
                sizeFactor = 1.0 - (((eb['size']-500) / 500.0) * circularityFactor)
            else:
                sizeFactor = 1.0       
                
        # Adapt size factor depending on circularity confidence?
        
        # Calculate confidence with weightings applied
        # Brightness and quality of stain are used because they may preclude judgement
        # Circularity, numconsidered, and size may depend on polary; brightness and quality shouldn't.
        capability = (brightnessFactor * brightnessWeight) + (stainQuality * stainWeight)
        subWeights = (brightnessWeight + stainWeight)
        capability *= 1.0/subWeights


        # Get angleDifWeight also
        # Square weight to reward the more oppositional centroids
        angleDifWeight = 1.0 + (angleDif / 180.0)
        angleDifWeight *= angleDifWeight

        # Get average brightness ratio as additional weight
        # TODO Integrate this more fully; was a quick-fix for testing
        brightnessRatio = (eb['genBrightnessRatio'] + eb['genBrightnessRatio']) / 2.0
        if brightnessRatio > 0.5:
            if brightnessRatio > 0.8:
                brghtwght = (1.0 - brightnessRatio) / 0.2
            else:
                brghtwght = 1.0
        else:
            brghtwght = brightnessRatio / 0.5
        
        # REFACTORING NEEDED - propDist is the raw polarity score
        propDist = angleDifWeight * (percentDist / 100.0) * capability * circularityFactor * brghtwght

        # Get integer rating based on score & given thresholds (given at top of file)
        if propDist < ps2:
            if propDist < ps1:
                if propDist < ps0:
                    polarityScore = 0
                else:
                    polarityScore = 1 
            else:
                polarityScore = 2
        else:
            polarityScore = 3   

        
        # NOTE Need new way of getting numConsideredFactor weighting which doesn't rely on polarity score
        numConsideredFactor = 1
        numConsideredProportion = (eb['polNumConsidered'] / float(eb['pixelArea'])) + (eb['genNumConsidered'] / float(eb['pixelArea']))
        if numConsideredProportion < 0.15:
            numConsideredFactor = float((polarityScore - 1) / 3.0)
        elif numConsideredProportion < 0.25:
            numConsideredFactor = (3 - abs(polarityScore - 3)) / 3.0
        elif numConsideredProportion < 0.4:
            numConsideredFactor = (3 - abs(polarityScore - 2)) / 3.0
        else:
            numConsideredFactor = float(1 - ((polarityScore-1)  / 3.0))
        
        confidence = (capability * (subWeights/1.0)) + (numConsideredFactor * numConsideredWeight) + (circularityFactor * circularityWeight) + (sizeFactor * sizeWeight)

        #confidenceStr = "%.2f" % (confidence)
#        print "bri: " + str(brightnessFactor) + " , size: " + str(sizeFactor) + ", density: " + str(numConsideredFactor) + ", stain: " + str(stainQuality) + ", circularity: " + str(circularityFactor) + ": gives confidence " + confidenceStr

        
        # Ave brightness ratio:
        aveBrightnessRatio = (eb['genBrightnessRatio'] + eb['genBrightnessRatio']) / 2.0

        # Write to CSV
        fields = (capability, angleDif, angleDifWeight, percentDist/100.0, propDist, brightnessFactor, stainQuality, eb['genBrightness'], eb['polBrightness'], aveBrightnessRatio, pixelDistCoB, pixelDistCenter, sizeFactor, numConsideredFactor, circularityFactor, polarityScore, confidence)
        output_string = filename + "eb-" + str(eb['id']) + ",%.2f,%.1f,%.2f,%.2f,%.3f,%.2f,%.2f,%d,%d,%.2f,%.4f, %.4f,%.2f,%.2f,%.2f,%d,%.2f\n" % fields
        
        # Create the output directory if not exists
        if not os.path.exists(outputDir):
            os.makedirs(outputDir)
            csv = open(outputDir + "full-set.csv", "a")
            # Write header line
            csv.write("eb ID,capability,angleDif,angleDifWeight,percentDist,weightedDist,brightnessFactor,stainQuality,maxGenBrightness,maxPolBrightness,aveBrightnessRatio,pixelDistCoB,pixelDistCenter,sizeFactor,numConsideredFactor,circularityFactor,polarityRating,confidence\n")
            csv.close()
        
        # Write to full set file first
        csv = open(outputDir + "full-set.csv", 'a')
        csv.write(output_string)
        csv.close()
        
        # Then write to marker-specific file
        thisDir = outputDir + "Unknown\\"
        
        if "MOMP" in filename or "Momp" in filename or "momp" in filename:            
            if "OmcB1" in filename:
                thisDir = outputDir + "OmcB1_MOMP\\"
            elif "OmcB2" in filename:
                thisDir = outputDir + "OmcB2_MOMP\\"
            elif "Ctad1" in filename:
                thisDir = outputDir + "Ctad1_MOMP\\"
            else:
                thisDir = outputDir + "MOMP\\"
        elif "OmcB" in filename:
            if "OmcB1" in filename:
                thisDir = outputDir + "OmcB1\\"
            elif "OmcB2" in filename:
                thisDir = outputDir + "OmcB2\\"
            elif "OmcB3" in filename:
                thisDir = outputDir + "OmcB3\\"
            else:
                thisDir = outputDir + "OmcB\\"
        elif "LPS" in filename:
            thisDir = outputDir + "LPS\\"
        elif "Hsp60" in filename:
            thisDir = outputDir + "Hsp60\\"
        elif "TepP" in filename:
            thisDir = outputDir + "TepP\\"
        elif "Ctad1" in filename:
            thisDir = outputDir + "Ctad1\\"
        elif "Cdsf" in filename:
            thisDir = outputDir + "Cdsf\\"
        elif "PmpD" in filename:
            thisDir = outputDir + "PmpD\\"
        
        if not os.path.exists(thisDir):
            os.makedirs(thisDir)
        saveMarkerSpecific(thisDir, filename, output_string)
        
        
def saveMarkerSpecific(directory, filename, output):
    if "CF" in filename:
        csv = open(directory + "cell-free.csv", "a")
    elif "cell" in filename or "Cell" in filename:
        csv = open(directory + "cell.csv", "a")
    else:
        csv = open(directory + "egress.csv", "a")
    csv.write(output)
    csv.close()
    return


## todo
def checkStainQuality(gen, pol):
    cumulative = getCumulativeStack(gen) / len(gen)

    temp = Image.fromarray(cumulative)
    if convertFlag:
        temp = temp.convert('I')
    gray = temp.convert('L')
    
    #NOTE: Should equalise here

    thresholded = np.asarray(gray).copy()
    
    thresholded = equalise(thresholded)
    if thresholded is None:
        print "NOT DOUBLE STAINED - FAILED ON GEN BRIGHTNESS"
        return False, 0

    # First count the number of EB pixels
    thresholded[thresholded < detectionThresholdVal] = 0
    thresholded[thresholded >= detectionThresholdVal] = 1
    genEBSum = np.sum(thresholded)

    # Next count number of non-zero pixels which aren't in EB threshold
    thresholded[thresholded > 0] = 1
    thresholded[thresholded >= detectionThresholdVal] = 0
    genBGSum = np.sum(thresholded);

    ## Same as above for pol channel
    cumulative = getCumulativeStack(pol) / len(pol)
    temp = Image.fromarray(cumulative)
    if convertFlag:
        temp = temp.convert('I')
    gray = temp.convert('L')
    thresholded = np.asarray(gray).copy()
    thresholded = equalise(thresholded)
    if thresholded is None:
        print "NOT DOUBLE STAINED - FAILED ON POL BRIGHTNESS"
        return False, 0

    thresholded[thresholded < detectionThresholdVal] = 0
    thresholded[thresholded >= detectionThresholdVal] = 1
    polEBSum = np.sum(thresholded)

    thresholded[thresholded > 0] = 1
    thresholded[thresholded >= detectionThresholdVal] = 0
    polBGSum = np.sum(thresholded)

    # Next assess stain quality, where 1 is perfect, 0 is terrible
    # Check proportion of non-EB pixels which are non-black
    # A high proportion is bad; low proportion is good
    genQuality = float(genBGSum) / (262144.0 - genEBSum)
    polQuality = float(polBGSum) / (262144.0 - polEBSum)
    stainQuality = (1.0 - genQuality) * (1.0 - polQuality)

    print "stainQuality is " + str(stainQuality)

    doubleStained = True
    # First check if double stained by checking if either is less than 1%
    if genEBSum > polEBSum and polEBSum < (genEBSum * 0.01) and stainQuality > 0.25:
        doubleStained = False
    elif polEBSum > genEBSum and genEBSum < (polEBSum * 0.01) and stainQuality > 0.25:
        doubleStained = False

    return doubleStained, stainQuality

convertFlag = False

def processFolder(folder):
    # defines directory name as folder drag-dropped in (indexed as second 'argument')
    workDir = folder + "\\"
    #workDir = testDir

    nameArr = workDir.split('\\')
    filename = nameArr[len(nameArr)-4] + "-" + nameArr[len(nameArr)-3] + "-" + nameArr[len(nameArr)-2] + "-"

    # saveDir adds folder 'analysis' to the workDir that was dragged
    saveDir = workDir + "analysis\\"
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)

    # Want regex to recognise string of form '[anything]z[num][num]_ch00.tif'
    gChannel = glob.glob(workDir + "*z[0-9]*_ch00.tif")
    rChannel = glob.glob(workDir + "*z[0-9]*_ch01.tif")
    bChannel = glob.glob(workDir + "*z[0-9]*_ch02.tif")

    genChannel = gChannel
    polChannel = rChannel if len(rChannel) else bChannel
    
    

    if "swap" in filename:
        print "Swapped Folder"
        genChannel = polChannel
        polChannel = gChannel
    
    # Check if it's in 16 bit
    # TODO Convert appropriately!
    curImg = Image.open(genChannel[0])
    if curImg.mode == "I;16":
        print "16 BIT IMAGES, SKIPPING"
        return False
        convertFlag = True
    else:
        convertFlag = False
            
    
    if len(rChannel) and len(bChannel) and len(gChannel):
        print "TRIPLE STAINED"
        return False


    # defines equalisedStack as result of getEqualisedStack function i.e. average of stack luminosity in green channel. Similar to a max projection
    equalisedStack = getEqualisedStack(genChannel)
    if equalisedStack is None:
        print "Stack below brightness threshold! Disregarding"
        return False

    # First check to ensure successful stains
    doubleStained, stainQuality = checkStainQuality(genChannel, polChannel)

    if not doubleStained:
        print "Not doublestained"
        
        if "MOMP" in filename or "Momp" in filename or "momp" in filename:     
            if "OmcB1" in filename:
                nonDoubled['omcb1momp'] += 1
            elif "OmcB2" in filename:
                nonDoubled['omcb2momp'] += 1
            elif "Ctad1" in filename:
                nonDoubled['ctad1momp'] += 1
            else:       
                nonDoubled['momp'] += 1
        elif "OmcB" in filename:
            if "OmcB1" in filename:
                nonDoubled['omcb1'] += 1
            elif "OmcB2" in filename:
                nonDoubled['omcb2'] += 1
            elif "OmcB3" in filename:
                nonDoubled['omcb3'] += 1
            else:
                nonDoubled['omcb'] += 1
        elif "LPS" in filename:
            nonDoubled['lps'] += 1
        elif "Hsp60" in filename:
            nonDoubled['hsp60'] += 1
        elif "TepP" in filename:
            nonDoubled['tepp'] += 1
        elif "Ctad1" in filename:
            nonDoubled['ctad1'] += 1
        elif "Cdsf" in filename:
            nonDoubled['cdsf'] += 1
        elif "PmpD" in filename:
            nonDoubled['pmpd'] += 1
        
        #f = open(errFile, 'a')
        #f.write(nameArr[len(nameArr)-3] + "/" + nameArr[len(nameArr)-2] + " not double stained!\n")
        #f.close()
        return False

    # Uses the 'max projection' from getequalisedstack within the getcandidateEBs function, to locate EBs in the image. Defines each as 'ebs' and prints for user
    ebs = getCandidateEBs(equalisedStack)

    # Calls function to find brightest slice per channel using EBs, which adds this info to the dictionary and prints it for user usage
    ebs = findEachBrightestSlice(genChannel, polChannel, ebs)

    # Crops EBs, finds CoB, distance between CoB and CoI, and finds angle between them (utilises other functions like findCentre)
    ebs = processEBs(genChannel, polChannel, ebs)

    # Print output, save files etc.
    generateOutput(filename, saveDir, ebs, stainQuality)

    if "MOMP" in filename or "Momp" in filename or "momp" in filename:     
        if "OmcB1" in filename:
            numStack['omcb1momp'] += 1
        elif "OmcB2" in filename:
            numStack['omcb2momp'] += 1
        elif "Ctad1" in filename:
            numStack['ctad1momp'] += 1
        else:       
            numStack['momp'] += 1
    elif "OmcB" in filename:
        if "OmcB1" in filename:
            numStack['omcb1'] += 1
        elif "OmcB2" in filename:
            numStack['omcb2'] += 1
        elif "OmcB3" in filename:
            numStack['omcb3'] += 1
        else:
            numStack['omcb'] += 1
    elif "LPS" in filename:
        numStack['lps'] += 1
    elif "Hsp60" in filename:
        numStack['hsp60'] += 1
    elif "TepP" in filename:
        numStack['tepp'] += 1
    elif "Ctad1" in filename:
        numStack['ctad1'] += 1
    elif "Cdsf" in filename:
        numStack['cdsf'] += 1
    elif "PmpD" in filename:
        numStack['pmpd'] += 1

    return True


## MAIN CODE ###################################################################

if len(sys.argv) == 2:
    sourceFolder = sys.argv[1]
else:
    sourceFolder = testDir

NumberFailed = 0

NumberSucceeded = 0


# Dictionaries for counting numberfailed etc.
# TODO Clean up...
nonDoubled = {}

nonDoubled['momp'] = 0

nonDoubled['omcb1momp'] = 0
nonDoubled['omcb2momp'] = 0
nonDoubled['ctad1momp'] = 0

nonDoubled['omcb1'] = 0
nonDoubled['omcb2'] = 0
nonDoubled['omcb3'] = 0
nonDoubled['omcb'] = 0
nonDoubled['lps'] = 0
nonDoubled['hsp60'] = 0
nonDoubled['tepp'] = 0
nonDoubled['ctad1'] = 0
nonDoubled['cdsf'] = 0
nonDoubled['pmpd'] = 0

numStack = {}

numStack['momp'] = 0

numStack['omcb1momp'] = 0
numStack['omcb2momp'] = 0
numStack['ctad1momp'] = 0

numStack['omcb1'] = 0
numStack['omcb2'] = 0
numStack['omcb3'] = 0
numStack['omcb'] = 0
numStack['lps'] = 0
numStack['hsp60'] = 0
numStack['tepp'] = 0
numStack['ctad1'] = 0
numStack['cdsf'] = 0
numStack['pmpd'] = 0


f = open(errFile, 'a')
f.write("//// Processing " + sourceFolder + " at " + time.strftime("%c") + " ////\n\n")
f.close()

for root, dirnames, filenames in os.walk(sourceFolder):
    
    # Temporary fix to ignore particular files in data set
    if "2016.12.07" not in root and "2016.12.1" not in root and "2016.12.2" not in root and "2017." not in root:
        continue
    if "Representatives" in root:
        continue
        
    
    dirPath = root.split('\\')
    if dirPath[len(dirPath) - 1] == "analysis":
        continue


    if len(filenames) > 0:
        print "Processing " + dirPath[len(dirPath) - 4] + "/" + dirPath[len(dirPath) - 3] + "/"+ dirPath[len(dirPath) - 2] + "/" + dirPath[len(dirPath) - 1]
        try:
            if processFolder(root):
                print "Ta-dah!\n"
                NumberSucceeded = NumberSucceeded + 1
            else:
                print "Uh-oh\n"
                NumberFailed = NumberFailed + 1
                failedStainCount += 1
                f = open(errFile, 'a')
                f.write(" - " + root + " failed\n\n")
                f.close()
        except Exception, e:
            exc_type, exc_value, exc_tb = sys.exc_info()
#
            print "Uh-oh - exception!\n"
            print str(e) + " at line " + str(exc_tb.tb_lineno)

            NumberFailed = NumberFailed + 1

f = open(errFile, 'a')
f.write("NON-DOUBLE STAINED:\n")
for stain in nonDoubled:
    f.write("- %s: %d out of %d\n" % (stain, nonDoubled[stain], (numStack[stain]) + nonDoubled[stain]))
#f.write("Total failed: " + str(NumberFailed) + "\n")
#f.write("Total non double-stained stacks: " + str(failedStainCount) + "\n")
f.write("\n////\n\n")
f.close()

#    for filename in fnmatch.filter(filenames, '*.tif'):
#    matches.append(os.path.join(root, filename))

print "\n\nRun successful - press any key to close this window"
print "Number failed = " + str(NumberFailed)
print "Number succeeded = " + str(NumberSucceeded)

#Freq = 1000 # Set Frequency To 2500 Hertz
#Dur = 100 # Set Duration To 1000 ms == 1 second
# Beep for completion :)
if NumberSucceeded == 0:
    winsound.Beep(1250,150)
    winsound.Beep(777,300)
else:
    winsound.Beep(1000,100)
    winsound.Beep(1500,200)    


# Wait for user input before closing window
test = raw_input()
