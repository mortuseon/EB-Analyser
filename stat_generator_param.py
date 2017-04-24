import sys
import os
import glob
import csv

#outputDir = "C:\Users\Hanne Stawarz\Documents\Homework\UCL Giorgio\Y4\The Polar Express\EB-Analyser\output"
outputDir = sys.argv[1]

#Thresholds
pol1 = 0.06
pol2 = 0.10
pol3 = 0.16

"""
Parses CSV files output by eb_analyser and aggregates stats in a new CSV file
in each folder; e.g. mean polarity rating, rating distribution...

ALSO: Applies new parameters to re-weight polarity score
      Use this for testing to find most appropriate weightings and parameters

"""


def processFolder(folder):

    csvFiles = glob.glob(folder + "\\*.csv")
    
    statsFile = open(folder + "\\stats-custom.csv", 'wb')
    
    totRatingAve = 0
    totConfAve = 0
    totCount = 0
    totScoreAve = 0
    totScoreWAve = 0
    totCapAve = 0
    
    
    for file in csvFiles:
        # Ignore our output file
        if "stats" in file:
            continue
    
                
        scoreCount = [0, 0, 0, 0]
        confidence = [0, 0, 0, 0]
                
        header = file.split("\\")
        header = header[len(header) - 1].replace(".csv", "")
        header += ",1,2,3,4,tot\n"
        statsFile.write(header)
        
        with open(file, "rb") as thisFile:
            content = csv.reader(thisFile, delimiter=",")
            num = 0
            
            aveRating = 0
            aveScore = 0
            wAveScore = 0
            aveConf = 0
            aveCap = 0


            for row in content:
                if str(row[0]) == "eb ID":
                    continue

                # Apply new weights to score
                angleDif = float(row[2])
                angleDifWeight = 1.0 + (angleDif / 180.0)
                angleDifWeight *= angleDifWeight

                cap = float(row[1])            
                
                circularity = float(row[15])    
                
                percentDist = float(row[4])
                
                brightnessRatio = float(row[10])
                
                if brightnessRatio > 0.5:
                    if brightnessRatio > 0.8:
                        brightnessWeight = (1.0 - brightnessRatio) / 0.2
                    else:
                        brightnessWeight = 1.0
                else:
                    brightnessWeight = brightnessRatio / 0.5
                
                
                score = angleDifWeight * percentDist * cap * circularity * brightnessWeight


                conf = float(row[len(row)-1])


                # Get rating based on score
                if score < pol3:
                    if score < pol2:
                        if score < pol1:
                            rating = 0
                        else:
                            rating = 1 
                    else:
                        rating = 2
                else:
                    rating = 3               
                
                scoreCount[rating] += 1
                confidence[rating] += conf
                
                aveRating += (rating + 1)

                aveScore += score
                aveConf += conf
                wAveScore += (cap * score)
                aveCap += cap
                num += 1

            if(num == 0):
                continue

            totRatingAve += aveRating
            totScoreAve += aveScore
            totScoreWAve += wAveScore
            totConfAve += aveConf
            totCapAve += aveCap
            totCount += num

            aveRating /= float(num)
            aveScore /= float(num)
            wAveScore /= float(num)
            aveConf /= float(num)
            aveCap /= float(num)
            
            
            proportion = [0,0,0,0]
            for i in range(0,4):
                if scoreCount[i] > 0:
                    confidence[i] /= scoreCount[i]
                proportion[i] = (scoreCount[i] / float(num)) * 100
            
            
            total = "Count," + ','.join(str(x) for x in scoreCount) + "," + str(num) + "\n"
            prop = "%," + ",".join("%.1f" % x for x in proportion) + ",100\n"
            conf = "Confidence," + ",".join("%.2f" % x for x in confidence) + ",-\n"
            rateLine = "Ave Rating,%.2f\n" % aveRating
            aveLine = "Ave Score,%.3f,Ave Conf:,%.2f\n" % (aveScore, aveConf)
            wAveLine = "w. Ave Score,%.3f,w. Ave Cap:,%.2f\n\n" % (wAveScore, aveCap)
            
            statsFile.write(total)
            statsFile.write(prop)
            statsFile.write(conf)
            statsFile.write(rateLine)
            statsFile.write(aveLine)
            statsFile.write(wAveLine)
    
    totScoreAve /= float(totCount)
    totScoreWAve /= float(totCount)
    totRatingAve /= float(totCount)
    totConfAve /= float(totCount)
    totCapAve /= float(totCount)
        
    statsFile.write("AVE RATING:,%.3f,AVE CONF:,%.2f,\n" % (totRatingAve, totConfAve))
    statsFile.write("AVE SCORE:,%.3f, w. AVE SCORE:,%.3f,AVE CAP:,%.2f,\n" % (totScoreAve, totScoreWAve, totCapAve))
    print " > AVERAGE RATING: %.3f" % totRatingAve
    statsFile.close()
        


for root, dirnames, filenames in os.walk(outputDir):

    dirPath = root.split('\\')
    if dirPath[len(dirPath) - 1] == "output":
        continue


    if len(filenames) > 0:
        print "Processing " + dirPath[len(dirPath) - 1]
        try:
            processFolder(root)
        except Exception, e:
            exc_type, exc_value, exc_tb = sys.exc_info()

            print "Uh-oh - exception!\n"
            print str(e) + " at line " + str(exc_tb.tb_lineno)

test = raw_input()    
