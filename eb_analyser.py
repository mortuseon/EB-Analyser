# -*- coding: utf-8 -*-
# import the packages we need
from PIL import Image, ImageDraw
import numpy as np
import sys
import math
import os
import glob
import time

#### TODO
# - Confidence
# - Polarity judgement

## USER VARIABLES ##############################################################

# Directory to be used whilst running tests
#testDir = "C:\Users\Hanne Stawarz\Documents\Homework\UCL Giorgio\Y4\Code\\"
#testDir = "C:\Users\Gulliver\Canopy\Code\B\\"
testDir = "C:\Users\Hanne Stawarz\Documents\Homework\UCL Giorgio\Y4\Code\Code_reworked\A\\"

csvFile = "D:\Gulliver\Projects\eb-analyser\master-analysis-03.csv"

errFile = "D:\Gulliver\Projects\eb-analyser\err.txt"

# extra boundary around box, as a scale factor (e.g. 1.5 = 50% larger, 2 = 100% larger)
boxMargin = 2

# defines minimum box size - EBs smaller than this no of pixels not considered
minBoxWidth = 4
maxBoxWidth = 16

# TODO: Calculate threshold intelligently...
#
# currently considers only pixels exceeding this brightness value (where 0 = black and 255 = white)
threshold = 200

# Threshold brightness for detecting EBs in binary image (e.g. pixels brighter than this will be set to 255, otherwise 0)
detectionThresholdVal = 30

# Threshold determining how bright an EB stain should be to be considered viable
aveBrightnessThreshold = 20 # used to be 20

# Minimum brightness for a stack to have to be considered viable, in range 0 - 255
minStackBrightness = 25 # used to be 25

failedStainCount = 0;

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
        if cumSum > threshold:
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
    for y in range(miny, maxy):
        for x in range(minx, maxx):
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
    img = img.convert('L')

    # converts to array, defines as imArr and equalises using prior function
    imArr = np.asarray(img)


    imArr = equalise(imArr)

    # TODO This is terrible; need better error handling
    if imArr is None:
        return None, None, None, None, None
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

    return rgbimg, cobX, cobY, dist, numConsidered

def getCandidateEBs(eqMaxproj):

    # Created (new) temporary image 'object' from the 2D array equalised stack ('max projection') and defines as 'temp'. Converts temp to luminosity only* could remove
    # Converts...back to array (grey) and copies it
    temp = Image.fromarray(eqMaxproj)
    gray = temp.convert('L')

    thresholded = np.asarray(gray).copy()

    # Creates blobs (easier for code to parse) by setting all pixels with brightness <100 as 0 (black) and all >100 as white (255)
    thresholded[thresholded < detectionThresholdVal] = 0
    thresholded[thresholded >= detectionThresholdVal] = 255
    #Image.fromarray(thresholded).show()
    ogThresholded = thresholded

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

                # value of yTemp is increased when a white pixel is found, i.e. moves down in image to check for brightness neighbours
                while thresholded[yTemp + 1, xTemp] == 255:

                    yTemp += 1

                    # Abort if you hit the bottom edge
                    if ((yTemp + 1) == 512):
                        break

                    # xTempLeft evaluates neighbours 1 pixel to the left of xTemp and will move only if it finds a white pixel. Helps to define leftmost boundary
                    # Then minx is set as the lowest x co-ordinate found by xTempLeft, i.e. the leftmost white pixel (left boundary)
                    # (Note that it doesn't bother if it is at an edge)
                    if (xTemp > 0):
                        xTempLeft = xTemp - 1
                        while thresholded[yTemp, xTempLeft] == 255:
                            if xTempLeft > 0:
                                xTempLeft -= 1
                            else:
                                break

                        if xTempLeft < minx:
                            minx = xTempLeft

                    # Does same as above except this is a temporary value scanning right, i.e. defines rightmost boundary and sets as maxX
                    if (xTemp < 511):
                        xTempRight = xTemp + 1
                        while thresholded[yTemp, xTempRight] == 255:
                            if xTempRight < 511:
                                xTempRight += 1
                            else:
                                break
                        if xTempRight > maxx:
                            maxx = xTempRight
                    # maxY is initially named as the lowest white pixel found thus far - needs to be verified
                maxy = yTemp

                #  Need to verify that we've found the lowest point (max y) of the EB
                #  If maxy == 512, then maxYFound = true, and we're at the bottom of the image; not possible to go deeper
                maxYFound = (maxy == 511)
                while not maxYFound:

                    # Move from left to right (minx to maxx) along this row
                    for i in range(minx, maxx + 1):
                        #  If you see any white, increment yTemp to look at next row, and start from the left-hand side again
                        if(thresholded[yTemp + 1, i] == 255):
                            yTemp += 1
                            break
                        # If you made it all the way to the right without seeing a white pixel, then the row above was the bottom row
                        elif (i == maxx):
                            maxYFound = True

                # Set maxy to new ytemp
                maxy = yTemp

                # Ignore if it's an edge boi
                if maxx == 511 or maxy == 511 or minx == 0:
                    print "Disregarding EB at edge of image, coordinates " + str(minx) + ", " + str(miny)
                    continue

                # Defines box size (width and height via x and y co-ordinates on either side). Radius via dividing by 2
                hDist = (maxx - minx) / 2
                vDist = (maxy - miny) / 2

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

                # If the actual EB isn't within 80% of the spherical shape, mark it non-spherical and disregard
                if(areaSum / estArea < 0.75):
                    print "Disregarding EB at " + str(minx) + ", " + str(miny) + " due to non-circular shape (may be overlap)"
                    continue

                # defined above as arbitrary minimum size of EB, below which identified spots are ignored/not boxed
                if halfBoxWidth < minBoxWidth:
                    print "Disregarding undersized candidate EB at coordinates " + str(minx) + ", " + str(miny)
                    continue

                if halfBoxWidth >= maxBoxWidth:
                    print "Disregarding oversized candidate EB"
                    continue

                # Store diameter before scaling box by margin
                ebDiam = halfBoxWidth * 4

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
                    print "Edge boyyy"
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
                thisEB['xMax'] = int(xOrigin + boxWidth)
                thisEB['yMax'] = int(yOrigin + boxWidth)
                thisEB['pixelSize'] = ebDiam
                thisEB['pixelArea'] = areaSum
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
    #rgbimg.show()

    return ebs

###
def findEachBrightestSlice(genSlices, polSlices, ebs):
    for eb in ebs:

        # curMaxBrightness keeps track of the greatest 'brightness' value seen in a slice for each channel
        curMaxBrightnessGen = 0
        curMaxBrightnessPol = 0

        # Saves the index of the brightest slice
        # Initialised to -1 just to identify a failed run; will be overwritten immediately. Not initialised as zero (zero may falsely appear as valid value)
        genBrightestSlice = -1
        polBrightestSlice = -1

        # iterates through each slice, length defined by number of z_00 etc
        for i in range (0, len(genSlices)):

            # Initialise sum of slice brightness to zero (general stain)
            curGenSum = 0;
            # Opens the general slice - [i] defines iteration
            curGenSlice = Image.open(genSlices[i])
            # Converts to luminosity
            curGenSlice = curGenSlice.convert('L')
            # Converts to array so calculations can be performed (array per slice)
            curGenSlice = np.asarray(curGenSlice)

            # same as above but in polar candidate channel
            curPolSum = 0;
            curPolSlice = Image.open(polSlices[i])
            curPolSlice = curPolSlice.convert('L')
            curPolSlice = np.asarray(curPolSlice)

            # Looks within box range (co-ordinates defined previously), adds pixel brightness to total box brightness per slice
            for x in range (eb['xMin'], eb['xMax']):
                for y in range (eb['yMin'], eb['yMax']):
                    curGenSum += curGenSlice[y, x]
                    curPolSum += curPolSlice[y, x]

            # curGenSum is sum of pixel values within EB box (gen stain) - if a higher brightness is found, this now becomes curMaxBrightnessGen
            # brightest slice is also set to 'i'
            if curGenSum > curMaxBrightnessGen:
                curMaxBrightnessGen = curGenSum
                genBrightestSlice = i

            # as above but for polar candidate channel
            if curPolSum > curMaxBrightnessPol:
                curMaxBrightnessPol = curPolSum
                polBrightestSlice = i

        # Creates new property in dictionary (the brightest slice index in a stack e.g. z_10) assigned from analysis above. For each stain
        eb['genMaxSlice'] = genBrightestSlice
        eb['genBrightness'] = curMaxBrightnessGen
        eb['polMaxSlice'] = polBrightestSlice
        eb['polBrightness'] = curMaxBrightnessPol

    return ebs

### This function finds the centre of brightness for each stain, as well as the distance and angle between the two
def processEBs(genChannel, polChannel, ebs):

    for eb in ebs:

        # Opens brightest slice in the general channel, indexed in genMaxSlice dictionary property
        gChanSlice = Image.open(genChannel[eb['genMaxSlice']])
        # Crops the brightest slice to the constraints set by the max/min co-ordinates (includes 'safety margin' of 1.5)
        gChanImg = gChanSlice.crop((eb['xMin'],eb['yMin'],eb['xMax'],eb['yMax']))
        # Saves cropped image as property of EB dictionary (unique to each EB)

        eb['genChanImg'] = gChanImg

        # as above but for polar candidate channel
        rChanSlice = Image.open(polChannel[eb['polMaxSlice']])
        rChanImg = rChanSlice.crop((eb['xMin'],eb['yMin'],eb['xMax'],eb['yMax']))
        eb['polChanImg'] = rChanImg

        eb['genAveBrightness'] = getAverageValue(np.asarray(gChanImg), gChanImg.size[0], gChanImg.size[1])

        # Adds processImg function output as dictionary definitions to dictionary 'eb', includes:
        #genChanImgProcessed is final processed general channel image
        #genCobX and Y are X and Y co-ordinates of the centre of brightness
        #genCoBDist is the distance between the centre of the cropped EB and the centre of brightness
        #genNumConsidered outputs how many pixels are considered when calculating the CoB, at the moment threshold is arbitrary but later this will be more informative (sorted threshold by descent)


        eb['genChanImgProcessed'], eb['genCoBX'], eb['genCoBY'], eb['genCoBDist'], eb['genNumConsidered'] = processImg(gChanImg)

        # TODO Need better error handling
        if eb['genChanImgProcessed'] is None:
            print "Failed on general channel processing of eb " + str(eb['id']) + "; too dark"
            eb['failed'] = True
            continue


        eb['polAveBrightness'] = getAverageValue(np.asarray(rChanImg), rChanImg.size[0], rChanImg.size[1])
        eb['polChanImgProcessed'], eb['polCoBX'], eb['polCoBY'], eb['polCoBDist'], eb['polNumConsidered'] = processImg(rChanImg)

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
        yDist = eb['genCoBX'] - eb['polCoBY']

        # Use pythag to calc distance between CoBs, and check that it's positive (force it to be positive)
        pixelDist = math.sqrt(math.pow(xDist, 2) + math.pow(yDist, 2))
        pixelDist = pixelDist if pixelDist > 0 else pixelDist * -1

        # Get pixel dist as a percentage of EB diameter
        percentDist = (pixelDist / eb['pixelSize']) * 100


        # Write CoB to file and percentage toooo
        f.write("  CoB Difference:   " + str(pixelDist) + " pixels (%.1f%% of eb width)\n\n" % percentDist)
        f.write("////////////////////////////////////////////////////////////////\n\n")
        f.close()

        # sliceDif
        sliceDif = int(math.fabs(eb['genMaxSlice'] - eb['polMaxSlice']))

        # Polarity Score
        # Depends on CoB distance only?
        # 0-10%: 0
        # 10-25%: 1
        # 25-40%: 2
        # 40%+: 3

        polarityScore = 0
        propDist = percentDist / 100.0
        # SliceDif?
        if propDist < 0.3:
            if propDist < 0.2:
                if propDist < 0.1:
                    polarityScore = 0
                else:
                    polarityScore = 1
            else:
                polarityScore = 2
        else:
            polarityScore = 3

        # eb['genNumConsidered'] and polNumConsidered, get them as percentages
        # numConsidered links to confidence; greater confidence if numConsidered
        # is high and polarity score is low and vice versa?
        # Confidence
        # Depends on brightness of stains, slice dif, background noise, CoB dist from geom center
        # Consider using spherical measure in later tests?
        # Also size above 600nm
        brightnessFactor = 1
        if eb['genBrightness'] < eb['polBrightness']:
            brightnessFactor = (eb['genBrightness'] / float(eb['polBrightness']))
        else:
            brightnessFactor = (eb['polBrightness'] / float(eb['genBrightness']))

        sizeFactor = 1
        # decrease linearly between 600 and 900
        if eb['size'] > 600 and eb['size'] < 1000:
            sizeFactor = max(0, 1.0-((eb['size'] - 600) / 400.0))

        # Get number considered for candidate polar stain relative to number of pixels in EB
        numConsideredFactor = 1
        numConsideredProportion = eb['polNumConsidered'] / float(eb['pixelArea'])
        if numConsideredProportion < 0.15:
            numConsideredFactor = float(polarityScore / 3.0)
        elif numConsideredProportion < 0.25:
            numConsideredFactor = (3 - abs(polarityScore - 2)) / 3.0
        elif numConsideredProportion < 0.4:
            numConsideredFactor = (3 - abs(polarityScore - 1)) / 3.0
        else:
            numConsideredFactor = float(1 - (polarityScore  / 3.0))

        confidence = (brightnessFactor + sizeFactor + numConsideredFactor + stainQuality) / 4
        confidenceStr = "%.1f" % (confidence)
        print str(brightnessFactor) + " * " + str(sizeFactor) + " * " + str(numConsideredFactor) + " * " + str(stainQuality) + " gives confidence " + confidenceStr

        # Write to CSV
        # Fields: File Name,Gen CoB Dist,Gen Max Slice,Pol CoB Dist,Pol Max Slice,CoB Difference,CoB Difference as %,Angle Difference,Slice Difference,Size,Direction,Polarity Score,Confidence
        csv = open(csvFile, 'a')
        csv.write(filename + "eb-" + str(eb['id']) + "," + str(eb['genCoBDist']) + "," + str(eb['genMaxSlice']) + "," + str(eb['polCoBDist']) + "," + str(eb['polMaxSlice']) + "," + str(pixelDist) + "," + str(round(percentDist,1)) + "," + str(round(angleDif,1)) + "," + str(sliceDif) + "," + str(eb['size']) + "," + str(eb['polarityDir']) + "," + str(polarityScore) + "," + confidenceStr + "\n")
        csv.close()

## todo
def checkStainQuality(gen, pol):
    cumulative = getCumulativeStack(gen) / len(gen)

    temp = Image.fromarray(cumulative)
    gray = temp.convert('L')

    thresholded = np.asarray(gray).copy()

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
    gray = temp.convert('L')
    thresholded = np.asarray(gray).copy()

    thresholded[thresholded < detectionThresholdVal] = 0
    thresholded[thresholded >= detectionThresholdVal] = 1
    polEBSum = np.sum(thresholded)

    thresholded[thresholded > 0] = 1
    thresholded[thresholded >= detectionThresholdVal] = 0
    polBGSum = np.sum(thresholded)

    # Next assess stain quality, where 1 is perfect, 0 is terrible
    # Check proportion of non-EB pixels which are non-black
    # A high proportion is bad; low proportion is good
    genQuality = genBGSum / (262144 - genEBSum)
    polQuality = polBGSum / (262144 - polEBSum)
    stainQuality = (1 - genQuality) * (1 - polQuality)

    print "stainQuality is " + str(stainQuality)

    doubleStained = True
    # First check if double stained by checking if either is less than half...
    if genEBSum > polEBSum and polEBSum < (genEBSum * 0.5) and stainQuality > 0.25:
        doubleStained = False
    elif polEBSum > genEBSum and genEBSum < (polEBSum * 0.5) and stainQuality > 0.25:
        doubleStained = False

    return doubleStained, stainQuality

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
    gChannel = glob.glob(workDir + "*z[0-9][0-9]_ch00.tif")
    rChannel = glob.glob(workDir + "*z[0-9][0-9]_ch01.tif")
    bChannel = glob.glob(workDir + "*z[0-9][0-9]_ch02.tif")

    genChannel = gChannel
    polChannel = rChannel if len(rChannel) else bChannel

    # defines equalisedStack as result of getEqualisedStack function i.e. average of stack luminosity in green channel. Similar to a max projection
    equalisedStack = getEqualisedStack(genChannel)
    if equalisedStack is None:
        print "Stack below brightness threshold! Disregarding"
        return False

    # First check to ensure successful stains
    doubleStained, stainQuality = checkStainQuality(genChannel, polChannel)

    if not doubleStained:
        print nameArr[len(nameArr - 2)] + " - Not double stained"
        f = open(errFile, 'a')
        f.write(nameArr[len(nameArr)-3] + "/" + nameArr[len(nameArr)-2] + " not double stained!\n")
        f.close()
        failedStainCount += 1
        return False

    # Uses the 'max projection' from getequalisedstack within the getcandidateEBs function, to locate EBs in the image. Defines each as 'ebs' and prints for user
    ebs = getCandidateEBs(equalisedStack)

    # Calls function to find brightest slice per channel using EBs, which adds this info to the dictionary and prints it for user usage
    ebs = findEachBrightestSlice(genChannel, polChannel, ebs)

    # Crops EBs, finds CoB, distance between CoB and CoI, and finds angle between them (utilises other functions like findCentre)
    ebs = processEBs(genChannel, polChannel, ebs)

    # Print output, save files etc.
    generateOutput(filename, saveDir, ebs, stainQuality)

    return True


## MAIN CODE ###################################################################

sourceFolder = sys.argv[1]

NumberFailed = 0
NumberSucceeded = 0

f = open(errFile, 'a')
f.write("//// Processing " + sourceFolder + " at " + time.strftime("%c") + " ////\n\n")
f.close()

for root, dirnames, filenames in os.walk(sourceFolder):

    dirPath = root.split('\\')
    if dirPath[len(dirPath) - 1] == "analysis":
        continue


    if len(filenames) > 0:
        print "Processing " + dirPath[len(dirPath) - 2] + "/" + dirPath[len(dirPath) - 1]
        try:
            if processFolder(root):
                print "Ta-dah!\n"
                NumberSucceeded = NumberSucceeded + 1
            else:
                print "Uh-oh\n"
                NumberFailed = NumberFailed + 1
                f = open(errFile, 'a')
                f.write(" - " + root + " failed\n\n")
                f.close()
        except Exception, e:
            exc_type, exc_value, exc_tb = sys.exc_info()

            print "Uh-oh - exception!\n"
            print str(e) + " at line " + str(exc_tb.tb_lineno)

            NumberFailed = NumberFailed + 1

f = open(errFile, 'a')
f.write("Total failed: " + str(NumberFailed) + "\n")
f.write("Total non double-stained stacks: " + str(failedStainCount) + "\n")
f.write("\n////\n\n")
f.close()

#    for filename in fnmatch.filter(filenames, '*.tif'):
#    matches.append(os.path.join(root, filename))

print "\n\nRun successful - press any key to close this window"
print "Number failed = " + str(NumberFailed)
print "Number succeeded = " + str(NumberSucceeded)

test = raw_input()
