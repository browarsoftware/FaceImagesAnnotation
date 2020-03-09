# Author: Tomasz Hachaj
# run first: face_align_all.py
# generates eigenfaces
# requires CelebA dataset: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
# import the necessary packages

import cv2
import numpy as np
import os
path = 'd:/Projects/Python/PycharmProjects/twarze_align'


files = []
how_many_images = 50000
variance_explained = 0.99

print('searching catalog images')
# r=root, d=directories, f = files
for r, d, f in os.walk(path):
    for file in f:
      files.append(os.path.join(r, file))

print(len(files))
a = 0
offset = 0
img = cv2.imread(files[0 + offset])
old_shape = img.shape
img_flat = img.flatten('F')

print('loading images')
T = np.zeros((img_flat.shape[0], how_many_images))
for i in range(how_many_images):
    if i % 1000 == 0:
        print(str(i) + " of " + str(how_many_images))
    img_help = cv2.imread(files[i + offset])
    T[:,i] = img_help.flatten('F') / 255

print('mean face')
mean_face = T.mean(axis = 1)

for i in range(how_many_images):
    T[:,i] -= mean_face

print('correlation matrix')
C = np.matmul(T.transpose(), T)
C = C / how_many_images

print('eigenvalues')
from scipy.linalg import eigh
w, v = eigh(C)
print('eigenvectors')
v_correct = np.matmul(T, v)

print('sorting')
sort_indices = w.argsort()[::-1]
w = w[sort_indices]  # puttin the evalues in that order
v_correct = v_correct[:, sort_indices]

w_percen = w / sum(w)
variance = 0
cooef_number = 0
while variance < variance_explained:
    variance += w_percen[cooef_number]
    cooef_number = cooef_number + 1

how_many_eigen = cooef_number
print("requires ", how_many_eigen, " components to get ", variance, " of variance.")


print('normalizing')
norms = np.linalg.norm(v_correct, axis=0)# find the norm of each eigenvector
v_correct = v_correct / norms

#change all eigenvectors to have first coordinate positive - optional
for i in range(v_correct.shape[1]):
    if v_correct[0,i] < 0:
        v_correct[:, i] = -1 * v_correct[:, i]


print('saving')
#save results
np.save("pca.res//T_st", T)
np.save("pca.res//v_st", v_correct)
np.save("pca.res//w_st", w)
np.save("pca.res//mean_face_st", mean_face)
np.save("pca.res//norms_st", norms)
np.save("pca.res//old_shape_st", np.asarray(old_shape))

start = 0
stop = how_many_eigen
v_correct_use = v_correct[:,start:stop]

w_correct_use = w[start:stop]

#image_to_code = T[:,50]
image_to_code = T[:,0]

result = np.matmul(v_correct_use.transpose(), image_to_code)
reconstruct = np.matmul(v_correct_use, result)

#result_features = (1 / np.sqrt(w_correct_use)) * result

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

print(min(reconstruct + mean_face))
v1 = np.array([1, 1, 1])
v2 = np.array([2, 4, 8])
print((1.0 / v2) * v1)
reconstruct2 = scale_and_reshape(reconstruct, mean_face, old_shape)
cv2.imshow('reconstructed',reconstruct2)

image_to_code2 = scale_and_reshape(image_to_code, mean_face, old_shape)
cv2.imshow('original',image_to_code2)

mean_face2 = scale_and_reshape(mean_face, None,old_shape)
cv2.imshow('mean_face',mean_face2)

fe1 = scale_and_reshape(v_correct_use[:,0] * norms[0], None,old_shape)
cv2.imshow('First eigenface',fe1)

fe2 = scale_and_reshape(v_correct_use[:,1] * norms[1], None,old_shape)
cv2.imshow('Second eigenface',fe2)


fe2 = scale_and_reshape(v_correct[:,how_many_images-1] * norms[how_many_images-1], None,old_shape)
cv2.imshow('Last eigenface',fe2)

print(T.shape)

cv2.waitKey(0)
cv2.destroyAllWindows()

