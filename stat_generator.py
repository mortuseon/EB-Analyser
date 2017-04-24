import sys
import os
import glob
import csv

outputDir = sys.argv[1]

"""
Parses CSV files output by eb_analyser and aggregates stats in a new CSV file
in each folder; e.g. mean polarity rating, rating distribution...
"""

def processFolder(folder):

    csvFiles = glob.glob(folder + "\\*.csv")
    
    
    statsFile = open(folder + "\\stats.csv", 'wb')
    
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
                rating = int(row[len(row)-2]) - 1
                score = float(row[len(row)-13])
                conf = float(row[len(row)-1])
                cap = float(row[len(row)-17])
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
