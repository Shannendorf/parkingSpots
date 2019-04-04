import numpy as np
import cv2
from matplotlib import pyplot as plt
import cv2
import tensorflow as tf

CATEGORIES = ["free", "busy"]
model = tf.keras.models.load_model('64x3-parking-CNN.model')

def prepare(img_array):
    IMG_SIZE = 50
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

# For testing purposes, print the location of the mouse when you leftclick on the video
def printLocation(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        print("{}, {}".format(x,y))


# #Capture from video feed:
# cap = cv2.VideoCapture(0)

cap = cv2.VideoCapture('Video-opnames Smart Parking/1553861675359.mp4') # alles vol
# cap = cv2.VideoCapture('Video-opnames Smart Parking/1553880837766.mp4') # Bijna leeg, 1 man vertrekt
# cap = cv2.VideoCapture('Video-opnames Smart Parking/1553877508109.mp4') # Zichtbare problemen

fps = cap.get(cv2.CAP_PROP_FPS)

#Every x seconds
interval = int(fps * 10) # Hiermee verander je de interval van hoe lang het duurt om footage te proberen

#TODO Add additional parking locations
#Car locations y1,y2,x1,x2
cL = [[220, 280, 0, 10], # Dit is niet echt onze fout tbh xD
      [220, 280, 10, 50],
      [225, 285, 40, 80],
      [225, 290, 70, 110],
      [225, 295, 130, 170],
      [225, 295, 170, 215],
      [230, 295, 220, 265],
      [230, 295, 260, 315],
      [230, 295, 350, 400],
      [230, 295, 405, 440],
      [230, 295, 450, 490],
      [225, 290, 490, 530],
      [220, 275, 560, 580],
      [215, 270, 590, 605]]

# Array om mee bij te houden welke plaatsen bezet zijn
pSpace = []
for car in cL:
    pSpace.append("")

framecount = 0
first = True

while(cap.isOpened()):
    ### Capture frame-by-frame
    ret, frame = cap.read()

    # Als het lezen van de frame is gelukt:
    if ret==True:

        # Only do this the first time
        if first:
            height, width = frame.shape[:2]
            print("Height: {}  Width: {}".format(height, width))
            print("Fps: {}".format(int(fps)))
            print("The interval is {} seconds".format(int(interval/fps)))
            first = False

        ### Our operations on the frame come here

        # Only do this every interval
        if framecount == 0:
            print("{} seconds of video footage has gone by".format(int(interval/fps)))
            #Voor alle aangegeven auto locaties
            for c in range(len(cL)):
                # Cut out a specific part of the image
                car = frame[cL[c][0]:cL[c][1], cL[c][2]:cL[c][3]]

                # Check if it has a car in it
                prediction = model.predict(prepare(car))
                pSpace[c] = CATEGORIES[int(prediction[0][0])]
                # print(pSpace)

                """WARNING: ONLY FOR TESTING"""
                # Show the image
                # plt.imshow(car)
                # plt.show()

        # Hier tekenen we nog wat shit op het scherm voor iedere parkeerplaats
        for c in range(len(cL)):

            # Teken vierkant op de plaats waar wordt gekeken
            cv2.rectangle(frame, (cL[c][2], cL[c][0]), (cL[c][3], cL[c][1]), (0, 255, 0), 3)

            # Teken rondje op vrije plaatsen
            if pSpace[c] == "free":
                cv2.circle(frame, ( (cL[c][2] + cL[c][3])//2, (cL[c][0] + cL[c][1])//2 ), 13, (0, 0, 255), -1)

        framecount = (framecount + 1) % interval

        # cv2.setMouseCallback("frame", printLocation)

        ### Display the resulting frame
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()