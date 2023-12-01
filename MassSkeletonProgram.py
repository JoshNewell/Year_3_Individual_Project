import cv2
import mediapipe as mp
import numpy as np
import pickle
import os


PATH = f"./Spliced_videos/"
videos = os.listdir(PATH)

for segment in videos:
    segmentPath = PATH + segment

    outHandPos = []
    frameNum = 0

    video = cv2.VideoCapture(segmentPath)
    ret, frame = video.read()

    mp_draw = mp.solutions.drawing_utils
    mp_hand = mp.solutions.hands

    with mp_hand.Hands(min_detection_confidence = 0.5,
                    min_tracking_confidence = 0.5) as hands:
        while ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame.flags.writeable = False
            result_hands = hands.process(frame)
            frame.flags.writeable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            frameInital = np.zeros([2, 21, 3], dtype = float)
            frameHands = np.full_like(frameInital, -1.0)
            i = 0
            if result_hands.multi_hand_landmarks: 
                for hand in result_hands.multi_hand_landmarks:
                    j = 0
                    for landmark in hand.landmark:
                        frameHands[i][j][0] = landmark.x
                        frameHands[i][j][1] = landmark.y
                        frameHands[i][j][2] = landmark.z
                        j += 1
                    i += 1
            outHandPos.append(frameHands)

            frameNum += 1
            ret, frame = video.read()

    outHandPos = np.array(outHandPos)
    with open(f"./A_To_Z_Skeletons/{segment}-handpos.pkl",'wb') as f:
        pickle.dump(outHandPos, f)

    video.release()
