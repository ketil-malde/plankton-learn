from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import SGD

import os

import config as C

def create_model():
    # Load Inception minus the final prediction layer
    base_model = InceptionV3(weights='imagenet',
                         include_top=False,
                         input_shape=(299, 299, 3))  # or 1?
    tcls = os.listdir(C.train_dir)
    vcls = os.listdir(C.val_dir)
    if len(tcls) != len(vcls):
        print('Error: mismatch between training and validation classes')
        exit -1

    # Add a new prediction layer
    tmp = GlobalAveragePooling2D()(base_model.output)
    predictions = Dense(len(tcls), activation='softmax')(tmp)
    model = Model(inputs=base_model.input, outputs=predictions)

    return model
