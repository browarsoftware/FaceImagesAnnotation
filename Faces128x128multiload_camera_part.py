# Author: Tomasz Hachaj
# perfoms face detection and annotation using NN, requires camera
# download dlib face descriptor from https://github.com/davisking/dlib-models
# import the necessary packages
from imutils.video import VideoStream
from imutils import face_utils
import datetime
import argparse
import imutils
import time
import dlib
import cv2
import numpy as np

#dlib install:
#python 3.6
#python -m pip install https://files.pythonhosted.org/packages/0e/ce/f8a3cff33ac03a8219768f0694c5d703c8e037e6aba2e865f9bae22ed63c/dlib-19.8.1-cp36-cp36m-win_amd64.whl#sha256=794994fa2c54e7776659fddb148363a5556468a6d5d46be8dad311722d54bfcf

import tensorflow.keras
import os
#disable warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import tensorflow as tf
tf.get_logger().setLevel('ERROR')


print(f"Tensor Flow Version: {tf.__version__}")
print(f"Keras Version: {tensorflow.keras.__version__}")
print()
print("GPU is", "available" if tf.test.is_gpu_available() else "NOT AVAILABLE")

features_names = ("X5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes","Bald", "Bangs",
         "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair", "Blurry", "Brown_Hair", "Bushy_Eyebrows",
         "Chubby", "Double_Chin", "Eyeglasses", "Goatee", "Gray_Hair", "Heavy_Makeup",
         "High_Cheekbones", "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard",
         "Oval_Face","Pale_Skin","Pointy_Nose", "Receding_Hairline",    "Rosy_Cheeks", "Sideburns",
         "Smiling", "Straight_Hair", "Wavy_Hair", "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick",
         "Wearing_Necklace", "Wearing_Necktie","Young")


import os
import psutil
process = psutil.Process(os.getpid())
print(process.memory_info().rss / (1024 * 1024))

from tensorflow import keras
from imutils.face_utils import rect_to_bb
from imutils.face_utils import FaceAligner

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('../ShapePredictors/shape_predictor_68_face_landmarks.dat')
fa = FaceAligner(predictor, desiredFaceWidth=128, desiredFaceHeight=128)

print("[INFO] camera sensor warming up...")
vs = VideoStream(0).start()
time.sleep(2.0)

from tensorflow.keras import backend as K
import gc
# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream, resize it to
    # have a maximum width of 400 pixels, and convert it to
    # grayscale
    print("****************************************************")
    frame = vs.read()
    image = np.copy(frame)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    print("[INFO] detecting rect...")
        
    rects = detector(gray, 2)
    # loop over the face detections
    for rect in rects:
        print("[INFO] procesing rect...")
        # extract the ROI of the *original* face, then align the face
        # using facial landmarks
        (x, y, w, h) = rect_to_bb(rect)
        faceAligned = fa.align(image, gray, rect)
        cv2.imshow("fa", faceAligned)
        img = np.expand_dims(faceAligned, axis=0)
        
        for a in features_names:
            model = keras.Sequential([
            keras.layers.Flatten(input_shape=(128, 128, 3)),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(2, activation='softmax')
            ])
            model.load_weights('./checkpoints/' + a + '/' + a + 'rgb_128x128_checkpoint')

            id_class = np.argmax(model.predict(img), axis=1)
            print(a + ":" + str(id_class[0]))
            del model
            K.clear_session()
            gc.collect()
        
    # show the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
