#import img_process
import numpy as np
import os
import json
import random
from img_process import crop, img_to_array
from PIL import Image
from keras import regularizers
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras.utils import np_utils, Sequence, to_categorical

batch_size = 10
num_epochs = 20

class MY_Generator(Sequence):
    def __init__(self, batch_size,img_dir,label_file):
        self.img_dir, self.labels = img_dir, label_file
        self.batch_size = batch_size
        self.Label = json.load(open('%s' % self.labels, 'r'))
        self.imgs = os.listdir("%s" % self.img_dir)
        self.i = 1
    def __len__(self):
        return int(208 // self.batch_size)
    
    def __getitem__(self, idx):
        batch_x = []
        batch_y = []
        for x in self.imgs[(idx * self.batch_size):((idx + 1) * self.batch_size)]:
          if x == '.DS_Store':
            continue
          if x in self.Label:
            print(x)
            label = self.Label[x]
            suit = np.zeros(4)
            suit[(label[0]-1)] = 0.5
            number = np.zeros(13)
            number[(label[1]-1)] = 0.5
            batch_y += [np.concatenate((suit,number), axis=None)]
            im = Image.open(os.path.join("%s" % self.img_dir, x))
            im = crop(im)
            im = img_to_array(im)
            batch_x += [im]
          else:
            pass
        print(self.i)
        self.i += 1
        return np.array(batch_x), np.array(batch_y)

train_generator = MY_Generator(batch_size,'data','dats.json')

model = Sequential()
model.add(Conv2D(64, kernel_size=5, padding='same', activation='relu', input_shape=(512,512,1,)))
#model.add(MaxPooling2D(pool_size=2, padding='same'))
model.add(Conv2D(64, kernel_size=3, padding='same', activation='sigmoid'))
model.add(MaxPooling2D(pool_size=2, padding='same'))
model.add(Conv2D(32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2, padding='same'))
#model.add(MaxPooling2D(pool_size=2, padding='same'))
model.add(Flatten())
model.add(Dense(200,activation='relu'))
model.add(Dense(17, activation='softmax'))
adam = optimizers.adam(lr=0.00001,amsgrad=True)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

model.fit_generator(generator=train_generator,
                                          steps_per_epoch= (208 // batch_size),
                                          epochs=num_epochs,
                                          verbose = 1,
                                          use_multiprocessing=True,
                                          workers = 4,
                                          max_queue_size = 32
                                          )
