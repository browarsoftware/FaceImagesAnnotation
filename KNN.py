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

#T = np.load("pca.res/T_st.npy")
#v_correct = np.load("pca.res/v_st.npy")
#v_correct = v_correct[:,0:500]
v_correct = np.load("pca.res/v_st99%variance.npy")
#v_correct = v_correct[:,0:3131]
w = np.load("pca.res/w_st.npy")
mean_face = np.load("pca.res/mean_face_st.npy")
#norms = np.load("pca.res/norms_st.npy")
old_shape = np.load("pca.res/old_shape_st.npy")
#how_many_images = T.shape[1]

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
#for a in range(empty_matrix.shape[0]):
for a in range(empty_matrix.shape[0]):
    if a % 100 == 0:
        print(str(a) + " of " + str(empty_matrix.shape[0]))
    img_help = cv2.imread(files[a + 180000])
    img_help = img_help.flatten('F') / 255
    img_help -= mean_face
    result = np.matmul(vct, img_help)
    empty_matrix[a, :] = result

#np.save("pca.res//knn_train_dataset", empty_matrix)
np.save("pca.res//knn_valid_dataset", empty_matrix)

'''
img_help = cv2.imread(files[0])
cv2.imshow('img_help',img_help)
img_help = img_help.flatten('F') / 255
img_help -= mean_face
result = np.matmul(v_correct.transpose(), img_help)
reconstruct = np.matmul(v_correct, result)

#result_features = (1 / np.sqrt(w_correct_use)) * result


reconstruct2 = scale_and_reshape(reconstruct, mean_face, old_shape)
cv2.imshow('reconstructed',reconstruct2)
cv2.waitKey(0)
cv2.destroyAllWindows()


'''



'''
X = [[0,0], [1,1], [2,5], [3,5]]
y = [0, 0, 1, 1]
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X, y)
print(neigh.predict([[0.7,0]]))
'''










'''
for a in range(20):
    fe1 = scale_and_reshape(v_correct[:,a] * norms[a], None,old_shape)
    cv2.imshow('Eigenface ' + str(a)  ,fe1)


variance_explained = 0.99 # 155 coeff
w_percen = w / sum(w)
variance = 0
cooef_number = 0
while variance < variance_explained:
    variance += w_percen[cooef_number]
    cooef_number = cooef_number + 1
print(cooef_number)



start = 3
stop = 3131 # variance 0.99
v_correct = v_correct[:,start:stop]

np.save("pca.res//v_st99%variance", v_correct)


cv2.waitKey(0)
cv2.destroyAllWindows()
'''