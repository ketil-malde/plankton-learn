from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import SGD

def create_model():
    # Load Inception minus the final prediction layer
    base_model = InceptionV3(weights='imagenet',
                         include_top=False,
                         input_shape=(299, 299, 3))  # or 1?

    # Add a new 84-class prediction layer
    tmp = GlobalAveragePooling2D()(base_model.output)
    predictions = Dense(60, activation='softmax')(tmp)
    model = Model(inputs=base_model.input, outputs=predictions)

    return model
