# Author: Tomasz Hachaj
# generates faces eligning
# requires CelebA dataset: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
# import the necessary packages
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import argparse
import imutils
import dlib
import cv2

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor and the face aligner
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
fa = FaceAligner(predictor, desiredFaceWidth=128, desiredFaceHeight=128)

path = 'd:/Projects/Python/PycharmProjects/img_align_celeba'
path_out = 'd:/Projects/Python/PycharmProjects/twarze_align'

import os
files = []
# r=root, d=directories, f = files
for r, d, f in os.walk(path):
    for file in f:
      files.append(os.path.join(r, file))

print(len(files))
# load the input image, resize it, and convert it to grayscale
#path = 'd:\\Projects\\python\\PycharmProjects\\img_align_celeba\\202567.jpg'

how_many_images = len(files)

import sys, math

start = 6
part = 7

p1 = math.floor((start * how_many_images) / part)
p2 = math.floor((start + 1) * (how_many_images) / part)
print("start " + str(p1))
print("stop " + str(p2))

indx_range = range(p1, p2)

for i in indx_range:
    print(i)
    image = cv2.imread(files[i])

    image = imutils.resize(image, width=800)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # show the original input image and detect faces in the grayscale
    # image
    #cv2.imshow("Input", image)
    rects = detector(gray, 2)

    # loop over the face detections
    for rect in rects:
        # extract the ROI of the *original* face, then align the face
        # using facial landmarks
        (x, y, w, h) = rect_to_bb(rect)
        #faceOrig = imutils.resize(image[y:y + h, x:x + w], width=128, height=128)
        faceAligned = fa.align(image, gray, rect)

        cv2.imwrite('../twarze_align/%06d.png' % (i), faceAligned)
        # display the output images
        #cv2.imshow("Original", faceOrig)
        #cv2.imshow("Aligned", faceAligned)
        #cv2.waitKey(0)