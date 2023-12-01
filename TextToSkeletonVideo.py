import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
import random
import math
from dijkstar import Graph, find_path

print("""
Text-To-Sign Skeleton Program
-------------------------------------
Curator: Josh Alexander Newell 2023
-------------------------------------
The program uses the letter data base located in 'A_To_Z_Skeletons', 
no other directories should be added to this folder inaddition to the 
ones that are there.

The current black-list letters are; J  Q  V
""")

DATABASEPATH = "./A_To_Z_Skeletons/"
VIDEOPATH = "./Generated_videos/"

characterBlackList = np.array(['J', 'Q', 'V'])

#Parse the database in the folder.
def LoadSkeletonDataBase(filePath):
    aToz = [f for f in os.listdir(filePath) if os.path.isdir(filePath + f)]
    database = {}

    for letter in aToz:
        samples = [f for f in os.listdir(filePath + letter + "/")]
        loadedSamples = []
        for sample in samples:
            PATH = filePath + letter + "/" + sample
            with open(PATH, 'rb') as f:
                loadedSamples.append(pickle.load(f))
        database[letter] = loadedSamples

    return database

#Combine two frame sequences into one
def CombineFrameSequence(firstSequence, secondSequence):
    result = np.concatenate((firstSequence, secondSequence), axis=0)
    return result

#Get a value that represents the distance between the two frames
#Will only be applied to full end and start frames
def GetTransitionCost(firstFrame, secondFrame):
    NUMVALUES = 84 # numhands x numLandmarks x numCoordinates(except the z)

    firstHold = np.array(firstFrame)
    secondHold = np.array(secondFrame)

    firstMeanX = np.mean(firstHold[:, :, 0])
    firstMeanY = np.mean(firstHold[:, :, 1])
    secondMeanX = np.mean(secondHold[:, :, 0])
    secondMeanY = np.mean(secondHold[:, :, 1])

    firstHold[:, :, 0] = firstHold[:, :, 0] - firstMeanX
    firstHold[:, :, 1] = firstHold[:, :, 1] - firstMeanY
    firstHold[:, :, 2] = 0
    secondHold[:, :, 0] = secondHold[:, :, 0] - secondMeanX
    secondHold[:, :, 1] = secondHold[:, :, 1] - secondMeanY
    secondHold[:, :, 2] = 0

    sumFirst = np.sum(np.absolute(firstHold))
    sumSecond = np.sum(np.absolute(secondHold))

    transitionCost = np.absolute(sumFirst - sumSecond) / NUMVALUES

    return transitionCost

#Generate extra frames between two sequences inorder to smooth the video
#Will only be applied to full frames
def InterpolateFrames(firstSequence, secondSequence):

    TRANSITIONFACTOR = 1800

    firstLast = firstSequence[-1]
    secondFirst = secondSequence[0]

    transitionCost = GetTransitionCost(firstLast, secondFirst)

    numInterpolatedFrames = round(transitionCost*TRANSITIONFACTOR) + 1

    if numInterpolatedFrames <= 1:
        finalInterpolation = CombineFrameSequence(firstSequence, secondSequence)
    else:
        stepSequence = (secondFirst - firstLast) / numInterpolatedFrames

        newSequence = []

        for i in range(1, numInterpolatedFrames):
            newSequence.append(firstLast + (i * stepSequence))

        newSequence = np.array(newSequence)

        firstToNew = CombineFrameSequence(firstSequence, newSequence)
        finalInterpolation = CombineFrameSequence(firstToNew, secondSequence)

    return finalInterpolation

#Turn a frame sequence into a video
def FramesToVideo(frameSequence, path, videoName):

    width = 1024
    height = 1024

    video = cv2.VideoWriter(f"{path}{videoName}.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 10.0, (width, height))

    for frame in frameSequence:
        outFrame = np.zeros(shape=[width, height, 3], dtype=np.uint8)
        for hand in frame:
            for landmark in hand:
                xCoord = landmark[0]
                yCoord = landmark[1]
                if (xCoord >= 0) and (yCoord >= 0):
                    cv2.circle(outFrame, (int(xCoord*width), int(yCoord*height)), math.floor(0.005*width), (255, 0, 0), -1)
            
            cv2.line(outFrame, (int(hand[0][0]*width), int(hand[0][1]*height)), (int(hand[1][0]*width), int(hand[1][1]*height)), (255, 0, 0), 2)
            cv2.line(outFrame, (int(hand[1][0]*width), int(hand[1][1]*height)), (int(hand[2][0]*width), int(hand[2][1]*height)), (255, 0, 0), 2)
            cv2.line(outFrame, (int(hand[2][0]*width), int(hand[2][1]*height)), (int(hand[3][0]*width), int(hand[3][1]*height)), (255, 0, 0), 2)
            cv2.line(outFrame, (int(hand[3][0]*width), int(hand[3][1]*height)), (int(hand[4][0]*width), int(hand[4][1]*height)), (255, 0, 0), 2)
            cv2.line(outFrame, (int(hand[0][0]*width), int(hand[0][1]*height)), (int(hand[5][0]*width), int(hand[5][1]*height)), (255, 0, 0), 2)
            cv2.line(outFrame, (int(hand[0][0]*width), int(hand[0][1]*height)), (int(hand[17][0]*width), int(hand[17][1]*height)), (255, 0, 0), 2)
            cv2.line(outFrame, (int(hand[5][0]*width), int(hand[5][1]*height)), (int(hand[6][0]*width), int(hand[6][1]*height)), (255, 0, 0), 2)
            cv2.line(outFrame, (int(hand[6][0]*width), int(hand[6][1]*height)), (int(hand[7][0]*width), int(hand[7][1]*height)), (255, 0, 0), 2)
            cv2.line(outFrame, (int(hand[7][0]*width), int(hand[7][1]*height)), (int(hand[8][0]*width), int(hand[8][1]*height)), (255, 0, 0), 2)
            cv2.line(outFrame, (int(hand[5][0]*width), int(hand[5][1]*height)), (int(hand[9][0]*width), int(hand[9][1]*height)), (255, 0, 0), 2)
            cv2.line(outFrame, (int(hand[9][0]*width), int(hand[9][1]*height)), (int(hand[10][0]*width), int(hand[10][1]*height)), (255, 0, 0), 2)
            cv2.line(outFrame, (int(hand[10][0]*width), int(hand[10][1]*height)), (int(hand[11][0]*width), int(hand[11][1]*height)), (255, 0, 0), 2)
            cv2.line(outFrame, (int(hand[11][0]*width), int(hand[11][1]*height)), (int(hand[12][0]*width), int(hand[12][1]*height)), (255, 0, 0), 2)
            cv2.line(outFrame, (int(hand[9][0]*width), int(hand[9][1]*height)), (int(hand[13][0]*width), int(hand[13][1]*height)), (255, 0, 0), 2)
            cv2.line(outFrame, (int(hand[13][0]*width), int(hand[13][1]*height)), (int(hand[14][0]*width), int(hand[14][1]*height)), (255, 0, 0), 2)
            cv2.line(outFrame, (int(hand[14][0]*width), int(hand[14][1]*height)), (int(hand[15][0]*width), int(hand[15][1]*height)), (255, 0, 0), 2)
            cv2.line(outFrame, (int(hand[15][0]*width), int(hand[15][1]*height)), (int(hand[16][0]*width), int(hand[16][1]*height)), (255, 0, 0), 2)
            cv2.line(outFrame, (int(hand[13][0]*width), int(hand[13][1]*height)), (int(hand[17][0]*width), int(hand[17][1]*height)), (255, 0, 0), 2)
            cv2.line(outFrame, (int(hand[17][0]*width), int(hand[17][1]*height)), (int(hand[18][0]*width), int(hand[18][1]*height)), (255, 0, 0), 2)
            cv2.line(outFrame, (int(hand[18][0]*width), int(hand[18][1]*height)), (int(hand[19][0]*width), int(hand[19][1]*height)), (255, 0, 0), 2)
            cv2.line(outFrame, (int(hand[19][0]*width), int(hand[19][1]*height)), (int(hand[20][0]*width), int(hand[20][1]*height)), (255, 0, 0), 2)


        video.write(outFrame)

    video.release()

#Generate a frame sequence of the optinmum frame sequences to make the text as defined by the cost function.
#The cost function ca be:
#   'Random' (no consideration random sequences) - 1
#   'Linear' (Lowest weight from one letter to the next is the path) - 2
#   'Dynamic' (Takes whole sequence into consideration for shortest path) - 3
def SolveLetterNodeMesh(text, dataBase, costFunction, interpolate):
    caseText = text.upper()
    removedWhiteSpace = caseText.replace(" ", "")
    clean_string = np.array([s for s in removedWhiteSpace if s.isalnum() or s.isspace()])

    #Remove black listed characters
    for badLetter in characterBlackList:
        clean_string = np.delete(clean_string, np.where(clean_string == badLetter))

    #Apply calculation based on cost function
    #Random
    if costFunction == 1:
        finalSequenceList = []

        valOne = random.randrange(0, len(dataBase[clean_string[0]]))
        valTwo = random.randrange(0,len(dataBase[clean_string[1]]))
        finalSequenceList = [dataBase[clean_string[0]][valOne], dataBase[clean_string[1]][valTwo]]
        
        for i in range(2, len(clean_string)):
            valNext = random.randrange(0, len(dataBase[clean_string[i]]))
            finalSequenceList.append(dataBase[clean_string[i]][valNext])

    #Linear
    elif costFunction == 2:
        lowestCost = 1
        finalSequenceList = []

        for sequenceOne in dataBase[clean_string[0]]:
            for sequenceTwo in dataBase[clean_string[1]]:
                cost = GetTransitionCost(sequenceOne[-1], sequenceTwo[0])
                if cost < lowestCost:
                    lowestCost = cost
                    finalSequenceList = [sequenceOne, sequenceTwo]
        
        for i in range(2, len(clean_string)):
            nextBest = []
            lowestCost = 1
            for sequenceNext in dataBase[clean_string[i]]:
                cost = GetTransitionCost(finalSequenceList[-1][-1], sequenceNext[0])
                if cost < lowestCost:
                    lowestCost = cost
                    nextBest = sequenceNext
            finalSequenceList.append(nextBest)

    #Dynamic
    else:
        numNodes = 0
        nodeMeshIDs = []
        nodeMesh = Graph()

        #Add a node for each possable letter in each position
        for letter in clean_string:
            for possable in dataBase[letter]:
                numNodes += 1
                nodeMeshIDs.append(possable)

        #Add start and end nodes
        numNodes = numNodes + 2

        currentNode = 0

        for possable in range(0 + currentNode + 1, len(dataBase[clean_string[0]])+ currentNode + 1):
            nodeMesh.add_edge(currentNode, possable, 1)
        
        currentNode = 1
        letterStartNode = 1

        for letter in range(0, len(clean_string) - 1):
            for possable in range(0, len(dataBase[clean_string[letter]])):
                for connect in range(0, len(dataBase[clean_string[letter + 1]])):
                    cost = GetTransitionCost(dataBase[clean_string[letter]][possable][-1], dataBase[clean_string[letter + 1]][connect][0])
                    connectNum = connect + letterStartNode + len(dataBase[clean_string[letter]])
                    nodeMesh.add_edge(currentNode, connectNum, cost)
                currentNode += 1
            letterStartNode = currentNode

        for possable in range(0, len(dataBase[clean_string[-1]])):
            nodeMesh.add_edge(currentNode + possable, letterStartNode + len(dataBase[clean_string[-1]]), 1)

        path = find_path(nodeMesh, 0, numNodes - 1)

        sequenceIDs = np.array(path.nodes[1:-1]) - 1

        finalSequenceList = []

        for nodeID in sequenceIDs:
            finalSequenceList.append(nodeMeshIDs[nodeID])

    finalSequence = finalSequenceList[0]

    for sequence in range(1, len(finalSequenceList)):
        if interpolate:
            finalSequence = InterpolateFrames(finalSequence, finalSequenceList[sequence])
        else:
            finalSequence = CombineFrameSequence(finalSequence, finalSequenceList[sequence])

    return finalSequence

if __name__ == "__main__":
    dataBase = LoadSkeletonDataBase(DATABASEPATH)

    inputText = input("Please input text: ")

    costFunction = int(input("What cost function should be used (1: Random; 2: Linear; 3: Dynamic): "))

    videoTitle = input("Video name: ")

    sequence = SolveLetterNodeMesh(inputText, dataBase, costFunction, True)

    FramesToVideo(sequence, VIDEOPATH, videoTitle)







