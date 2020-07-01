# The imported generators expect to find training data in data/train
# and validation data in data/validation
from keras.models import load_model
from keras.callbacks import CSVLogger
from keras.optimizers import SGD

import os

from generators import train_generator, validation_generator
from create_model import create_model

import config as C

# Modify this to match last version saved
last = 0

def save_name(i):
    return ('models/epoch_'+str(i)+'.model')

def log(s):
    with open(C.logfile, 'a') as f:
        print(s, file=f)

if last == 0:
    log('Creating initial network from scratch.')
    if not os.path.exists('models'):
        os.makedirs('models')
    model = create_model()
else:
    model = load_model(save_name(last))

# Use log to file
logger = CSVLogger(C.logfile, append=True, separator='\t')

# we use SGD with a low learning rate
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9),
              loss='categorical_crossentropy',
              metrics=['mse', 'accuracy'])

def train_step(i):
    model.fit_generator(
        train_generator(C.train_dir), steps_per_epoch=1000, epochs=10,
        callbacks=[logger],
        validation_data=validation_generator(C.val_dir), validation_steps=500)
    model.save(save_name(i))


for i in range(last+1, last+10):
        log('Starting iteration '+str(i))
        train_step(i)
