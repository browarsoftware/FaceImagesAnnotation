# Author: Tomasz Hachaj
# run first: face_align_all.py
# performs training and validation of NN 40
# requires CelebA dataset: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
# import the necessary packages

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import DirectoryFunctions

print(tf.__version__)

import numpy as np
import os
path = 'd:\\Projects\\Python\\PycharmProjects\\twarze_align'
#PIL in pillow package
files = []
how_many_images_test = 180000
how_many_images_valid = 20000
epochs_count = 20

# r=root, d=directories, f = files
for r, d, f in os.walk(path):
    for file in f:
      files.append(os.path.join(r, file))

files = np.array(files)
from numpy import genfromtxt
my_data = genfromtxt('d:\\Projects\\Python\\PycharmProjects\\list_attr_celeba_align196095.txt',skip_header=1, delimiter=',', dtype=None, encoding=None)
features_names = ("X5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes","Bald", "Bangs",
         "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair", "Blurry", "Brown_Hair", "Bushy_Eyebrows",
         "Chubby", "Double_Chin", "Eyeglasses", "Goatee", "Gray_Hair", "Heavy_Makeup",
         "High_Cheekbones", "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard",
         "Oval_Face","Pale_Skin","Pointy_Nose", "Receding_Hairline",    "Rosy_Cheeks", "Sideburns",
         "Smiling", "Straight_Hair", "Wavy_Hair", "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick",
         "Wearing_Necklace", "Wearing_Necktie","Young")

my_data = np.floor((my_data[:, :] + 1) / 2 + 0.001)

feature_id = 0
import pandas as pd

for feature_id in range(40):
    print('*****************' + features_names[feature_id] + '*****************')
    print('train files')
    dict_test = {'filename' : files[0:how_many_images_test],
                 'class' : my_data[0:how_many_images_test,feature_id]}
    train_df = pd.DataFrame(dict_test)

    print('valid files')
    dict_valid = {'filename' : files[how_many_images_test:(how_many_images_test + how_many_images_valid)],
                  'class' : my_data[how_many_images_test:(how_many_images_test + how_many_images_valid),feature_id]}
    valid_help = my_data[how_many_images_test:(how_many_images_test + how_many_images_valid),feature_id]

    valid_df = pd.DataFrame(dict_valid)

    how_many_images_test = train_df.shape[0]
    how_many_images_valid = valid_df.shape[0]

    # The 1./255 is to convert from uint8 to float32 in range [0,1].
    image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    BATCH_SIZE = 32
    IMG_HEIGHT = 128
    IMG_WIDTH = 128

    print('image generator train')
    train_generator = image_generator.flow_from_dataframe(
            dataframe=train_df,
            directory='data/train',
            x_col="filename",
            y_col="class",
            target_size=(IMG_HEIGHT, IMG_WIDTH),
            batch_size=how_many_images_test,
            class_mode='raw',
            shuffle=False)
    image_batch_train, label_batch_train = next(train_generator)


    print('image generator valid')
    valid_generator = image_generator.flow_from_dataframe(
            dataframe=valid_df,
            directory='data/train',
            x_col="filename",
            y_col="class",
            target_size=(IMG_HEIGHT, IMG_WIDTH),
            batch_size=how_many_images_valid,
            class_mode='raw',
            shuffle=False)
    image_batch_valid, label_batch_valid = next(valid_generator)

    from tensorflow import keras
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(128, 128, 3)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(2, activation='softmax')
    ])

    import os
    os.environ["PATH"] += os.pathsep + "d:\\Program Files (x86)\\Graphviz2.38\\bin"
    tf.keras.utils.plot_model(
        model, to_file='model128x128.png', show_shapes=True, show_layer_names=True,
        rankdir='TB', expand_nested=True, dpi=256
    )


    model.summary()
    print('compiling model')
    model.compile(optimizer='Adamax',#adam, Adadelta, Adagrad, Adamax, FTRL, Nadam, RMSprop,SGD
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])


    # This function keeps the learning rate at 0.001 for the first ten epochs
    # and decreases it exponentially after that.
    def scheduler(epoch):
      if epoch < 20:
        return 0.001
      else:
        return 0.001 * tf.math.exp(0.1 * (10 - epoch))

    callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

    print('fiting model')
    model.fit(image_batch_train, label_batch_train,
              callbacks=[callback], epochs=epochs_count, batch_size = BATCH_SIZE)

    model.save_weights('./checkpoints/' + features_names[feature_id] + '/NEW__' + features_names[feature_id] + 'rgb_128x128_checkpoint')
    pc = model.predict_classes(image_batch_valid)
    print(pc.shape)
    DirectoryFunctions.append_line_to_file('./results/' + features_names[feature_id] + '.csv','id,predict,actual')
    for a in range(pc.shape[0] - 1):
        DirectoryFunctions.append_line_to_file('./results/' + features_names[feature_id] + '.csv',
            str(a) + "," + str(pc[a]) + "," + str(valid_help[a]))

    #clear memeory
    del dict_test
    del dict_valid
    del model
    from tensorflow.keras import backend as K
    K.clear_session()
    del train_generator
    del valid_generator
    del image_generator
    import gc
    gc.collect()
