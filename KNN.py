# Author: Tomasz Hachaj
# run first: GenerateEigenfacesForKNN.py
# generates features using eigenfaces
# requires CelebA dataset: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
# import the necessary packages

import cv2
import numpy as np

image_scale = 2
def scale(np1):
    np2 = (np1 - np.min(np1)) / np.ptp(np1)
    return np2

def scale_and_reshape(np1, mf, old_shape):
    if mf is None:
        np2 = np1.reshape(old_shape, order='F')
    else:
        np2 = (np1 + mf).reshape(old_shape, order='F')
    np2 = scale(np2)
    return np2

def nothing(x):
    pass

v_correct = np.load("pca.res/v_st99%variance.npy")
w = np.load("pca.res/w_st.npy")
mean_face = np.load("pca.res/mean_face_st.npy")
old_shape = np.load("pca.res/old_shape_st.npy")

vct = v_correct.transpose()
import os
path = 'd:\\Projects\\Python\\PycharmProjects\\twarze_align'
files = []
# r=root, d=directories, f = files
for r, d, f in os.walk(path):
    for file in f:
      files.append(os.path.join(r, file))


empty_matrix = np.zeros([180000, v_correct.shape[1]])
print(empty_matrix.shape[0])
for a in range(empty_matrix.shape[0]):
    if a % 100 == 0:
        print(str(a) + " of " + str(empty_matrix.shape[0]))
    img_help = cv2.imread(files[a])
    img_help = img_help.flatten('F') / 255
    img_help -= mean_face
    result = np.matmul(vct, img_help)
    empty_matrix[a, :] = result

np.save("pca.res//knn_train_dataset", empty_matrix)


empty_matrix = np.zeros([len(files) - 180000, v_correct.shape[1]])
print(empty_matrix.shape[0])
for a in range(empty_matrix.shape[0]):
    if a % 100 == 0:
        print(str(a) + " of " + str(empty_matrix.shape[0]))
    img_help = cv2.imread(files[a + 180000])
    img_help = img_help.flatten('F') / 255
    img_help -= mean_face
    result = np.matmul(vct, img_help)
    empty_matrix[a, :] = result

np.save("pca.res//knn_valid_dataset", empty_matrix)
