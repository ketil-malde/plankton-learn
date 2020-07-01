from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

def train_generator(tdir):
    return train_datagen.flow_from_directory(
        tdir,
        target_size=(299, 299),
        batch_size=32,
        class_mode='categorical')

def validation_generator(vdir):
    return test_datagen.flow_from_directory(
        vdir,
        target_size=(299, 299),
        batch_size=32,
        class_mode='categorical')
