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

cap = cv2.VideoCapture('Video-opnames Smart Parking/1553880837766.mp4')

fps = cap.get(cv2.CAP_PROP_FPS)

#Every x seconds
interval = int(fps * 60)

#TODO Add additional parking locations
#Car locations y1,y2,x1,x2
cL = [[230, 295, 210, 270],
      [230, 295, 165, 225],
      [230, 295, 10, 60]]

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
            for c in cL:
                # Cut out a specific part of the image
                car = frame[c[0]:c[1], c[2]:c[3]]

                # Check if it has a car in it
                prediction = model.predict(prepare(car))
                print(CATEGORIES[int(prediction[0][0])])

                # Show the image
                plt.imshow(car)
                plt.show()

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