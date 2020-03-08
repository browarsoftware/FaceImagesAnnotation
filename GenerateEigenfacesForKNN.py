import numpy as np
import cv2
import os
import DirectoryFunctions
# http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
#path = 'e:\\Projects\\python\\img_align_celeba'

#path = 'e:\\Projects\\python\\same_twarze'
path = 'd:/Projects/Python/PycharmProjects/twarze_align'
#path = 'd:/Projects/Python/PycharmProjects/same_twarze'

files = []
# r=root, d=directories, f = files
for r, d, f in os.walk(path):
    for file in f:
      files.append(os.path.join(r, file))

print('reading eigenvectors')
v_correct = np.load("pca.res/v_st.npy")
print('reading eigenvalues')
w = np.load("pca.res/w_st.npy")
print('reading mean face')
mean_face = np.load("pca.res/mean_face_st.npy")

variance_explained = 0.95 # 472 coeff
w_percen = w / sum(w)
variance = 0
cooef_number = 0
while variance < variance_explained:
    variance += w_percen[cooef_number]
    cooef_number = cooef_number + 1

print('coef number: ' + str(cooef_number))

how_many_eigen = cooef_number

start = 0
stop = how_many_eigen
v_correct = v_correct[:,start:stop]

w_correct_use = w[start:stop]
all_fetures = []

print('processing data')
for a in range(len(files)):
    if a % 1000 == 0:
        print(str(a) + " of " + str(len(files)))
    img_help = cv2.imread(files[a])
    img_help = img_help.flatten('F') / 255
    img_help -= mean_face

    result = np.matmul(v_correct.transpose(), img_help)
    result_features = result#(1 / np.sqrt(w_correct_use)) * result
    #result_features = (1 / np.sqrt(w_correct_use)) * result
    text_to_append = ''
    for b in range(result.shape[0]):
        if b > 0:
            text_to_append += ','
        text_to_append += str(result[b])
    DirectoryFunctions.append_line_to_file('eigen_features_dataset.csv', text_to_append)
