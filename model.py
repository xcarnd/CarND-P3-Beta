from keras.models import Sequential
from keras.layers.core import Dense, Lambda, Flatten, Activation, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
import csv
import cv2
import numpy as np

input_size = (160, 320, 3)
keep_rate = 0.5
batch_size = 32
steps_per_epoch = 4000 // batch_size
steps_per_validation = 1000 // batch_size
epochs = 50

def make_model(input_size, keep_rate):
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=input_size))
    model.add(Conv2D(6, kernel_size=(3, 3), activation='relu', padding='valid'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(keep_rate))
    model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', padding='valid'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(keep_rate))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='valid'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(keep_rate))
    model.add(Conv2D(64, kernel_size=(5, 5), activation='relu', padding='valid'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(keep_rate))
    model.add(Conv2D(64, kernel_size=(5, 5), activation='relu', padding='valid'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(keep_rate))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dense(10))
    model.add(Activation('relu'))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')
    return model

def read_datapoint_img(img_path, flip=False):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float)
    if flip:
        return np.fliplr(img)
    else:
        return img

def create_infinite_generator_with_list(source, batch_size):
    while True:
        batch_input = []
        batch_output = []
        while len(batch_input) < batch_size:
            nth = np.random.randint(len(source))
            img_path, steering, flip = source[nth]
            batch_input.append(read_datapoint_img(img_path, flip))
            batch_output.append(steering)
        yield (np.array(batch_input, dtype=np.float), np.array(batch_output, dtype=np.float))

def get_training_and_validation_generator(csv_path, training_set_size=0.8):
    full_set = []
    with open(csv_path, 'r') as f:
        c = csv.reader(f)
        # skip the header row
        next(c)
        for (center, left, right, steering, _, _, _) in c:
            # False means image do not need flipping before sending to the network, True otherwise
            full_set.append((center, float(steering), False))
            full_set.append((center, -float(steering), True))

    np.random.shuffle(full_set)
    num_training = int(len(full_set) * training_set_size)
    training, validation = full_set[:num_training], full_set[num_training:]
    print("Size of training set", len(training))
    print("Size of validation set", len(validation))
    return create_infinite_generator_with_list(training, batch_size), create_infinite_generator_with_list(validation, batch_size)

if __name__ == '__main__':
    t, v = get_training_and_validation_generator('/home/workspace/rec/driving_log.csv')
    model = make_model(input_size, keep_rate)
    model.summary()
    model.fit_generator(t, steps_per_epoch, epochs=epochs, validation_data=v, validation_steps=steps_per_validation) 
    model.save('model.h5')
    