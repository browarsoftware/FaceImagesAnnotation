# Author: Tomasz Hachaj
# run first: KNN.py
# validates KNN model
# requires CelebA dataset: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
# import the necessary packages


import numpy as np
import DirectoryFunctions
how_many_images_test = #INSERT HERE
how_many_images_valid = #INSERT HERE

from numpy import genfromtxt
print('loading data')
my_data = genfromtxt('d:\\Projects\\Python\\PycharmProjects\\list_attr_celeba_align196095.txt',skip_header=1, delimiter=',', dtype=None, encoding=None)

features_names = ("X5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes","Bald", "Bangs",
         "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair", "Blurry", "Brown_Hair", "Bushy_Eyebrows",
         "Chubby", "Double_Chin", "Eyeglasses", "Goatee", "Gray_Hair", "Heavy_Makeup",
         "High_Cheekbones", "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard",
         "Oval_Face","Pale_Skin","Pointy_Nose", "Receding_Hairline",    "Rosy_Cheeks", "Sideburns",
         "Smiling", "Straight_Hair", "Wavy_Hair", "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick",
         "Wearing_Necklace", "Wearing_Necktie","Young")

my_data = np.floor((my_data[:, :] + 1) / 2 + 0.001)

train = np.load("pca.res/knn_train_dataset.npy")
valid = np.load("pca.res//knn_valid_dataset.npy")

train = train[:,3:train.shape[1]]
valid = valid[:,3:valid.shape[1]]

from sklearn.neighbors import KNeighborsClassifier
for feature_id in range(len(features_names)):
    print(features_names[feature_id])
    y_train = my_data[0:how_many_images_test, feature_id]
    y_valid = my_data[how_many_images_test:(how_many_images_test + how_many_images_valid), feature_id]

    neigh = KNeighborsClassifier(n_neighbors=1, n_jobs = -1)
    print('fitting model')
    neigh.fit(train, y_train)
    print('predicting')
    pred = neigh.predict(valid)
    DirectoryFunctions.append_line_to_file('./knn.res/knn_' + features_names[feature_id] + '.csv', 'id,predict,actual')
    for a in range(pred.shape[0]):
        if a % 100 == 0:
            print(str(a) + " of " + str(pred.shape[0]))
        DirectoryFunctions.append_line_to_file('./knn.res/knn_' + features_names[feature_id] + '.csv',
            str(a) + "," + str(pred[a]) + "," + str(y_valid[a]))