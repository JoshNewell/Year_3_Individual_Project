import cv2
import mediapipe as mp
import numpy as np
import pickle

frameByFrame = False
exportValues = False
recording = False
another = True

print("""
Sign Video Splicing Program
-------------------------------------
Curator: Josh Alexander Newell 2023
-------------------------------------
'q' will quit the program.

'p' is used to pause the video in Continuos mode and will step the video in FrameByFrame mode. 

When in FrameByFrame mode the video can be recorded with 'r', once to start and again to stop.
When in Continuos mode the hand landmarks can be exported as a pickle file.

Exported files will be sent to the appropriate folder.
""")

while another:
    another = False
    frameIn = True
    exportIn = True
    frameNum = 0
    outHandPos = []

    doFrameByFrame = input("Run video frame by frame y/n: ")

    while frameIn:
        if doFrameByFrame == "y":
            frameIn = False
            frameByFrame = True

        elif doFrameByFrame == "n":
            frameIn = False
            frameByFrame = False

        else:
            doFrameByFrame = input("Please enter correct character y/n: ")

    if not frameByFrame:
        exportValuesIn = input("Export hand marker positions y/n: ")

        while exportIn:
            if exportValuesIn == "y":
                exportIn = False
                exportValues = True

            elif exportValuesIn == "n":
                exportIn = False
                exportValues = False

            else:
                exportValuesIn = input("Please enter correct character y/n: ")


    inputFile = input("Provide input file name: ")

    PATH = f"./Fingerspelling_videos/{inputFile}.mp4"

    video = cv2.VideoCapture(PATH)
    ret, frame = video.read()

    mp_draw = mp.solutions.drawing_utils
    mp_hand = mp.solutions.hands

    DRAW_LANDMARKS = True

    with mp_hand.Hands(min_detection_confidence = 0.5,
                    min_tracking_confidence = 0.5) as hands:
        while ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame.flags.writeable = False
            result_hands = hands.process(frame)
            frame.flags.writeable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            if recording:
                out.write(frame)

            if DRAW_LANDMARKS:
                if result_hands.multi_hand_landmarks:
                    for hand_landmark in result_hands.multi_hand_landmarks:
                        mp_draw.draw_landmarks(frame, hand_landmark, mp_hand.HAND_CONNECTIONS)

            cv2.imshow("Sign Video", frame)
            k = cv2.waitKey(1)

            if k == ord('q'):
                break

            if not frameByFrame:
                if k == ord('p'):
                    pause = True
                    while pause:
                        pK = cv2.waitKey(1)
                        if pK == ord('p'):
                            pause = False

            if frameByFrame:
                frameNext = True
                while frameNext:
                    fK = cv2.waitKey(1)
                    if fK == ord('p'):
                        frameNext = False

                    if fK == ord('r'):
                        if recording:
                            recording = False
                            out.release()
                            print("Recording Finished")
                        else:
                            width  = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
                            height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            comment = input("Recording comment: ")
                            out = cv2.VideoWriter(f"./Spliced_videos/{inputFile}-s{frameNum}-{comment}.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 10.0, (width, height))
                            recording = True
                            print("Recording Started")

            if exportValues:
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

    if exportValues:
        outHandPos = np.array(outHandPos) # Points are in form [frame][hand][landmark][xy], if anything doesn't exisit it will be value -1.0
        with open(f"./Hand_position_points/{inputFile}-handpos.pkl",'wb') as f:
            pickle.dump(outHandPos, f)

    video.release()
    if recording:
        recording = False
        out.release()
        print("Video complete finishing recording")
    cv2.destroyAllWindows()

    YN = input("Do another video y/n: ")

    runAgainIn = True

    while runAgainIn:
        if YN == "y":
            runAgainIn = False
            another = True

        elif YN == "n":
            runAgainIn = False

        else:
            YN = input("Please enter correct character y/n: ")
