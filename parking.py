from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import pickle
import time

# loading in the X and y variables created with the prepare_dataset.py
pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle","rb")
y = pickle.load(pickle_in)

X = X/255.0

# determines the amount/type of layers and amount of neurons used to train the model
# there will always be at least one convolutional layer with relu and maxpooling at the start and one dense layer with
# sigmoid at the end however
dense_layers = [0]
layer_sizes = [64]
conv_layers = [3]

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            # prints the names of the chosen layers and the time
            NAME = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer, layer_size, dense_layer, int(time.time()))
            print(NAME)

            # determines that the model is sequential
            model = Sequential()

            # adds the first convolutional layer of 3 by 3 with relu activation function and maxpooling of 2 by 2
            model.add(Conv2D(layer_size, (3, 3), input_shape=X.shape[1:]))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))

            # adds more conv_layers if determined
            for l in range(conv_layer-1):
                model.add(Conv2D(layer_size, (3, 3)))
                model.add(Activation('relu'))
                model.add(MaxPooling2D(pool_size=(2, 2)))

            # flattens the output of previous layer so that it can be used as input for the dense layer
            model.add(Flatten())

            # adds dense layers with relu activation function according to number of defined dense_layers
            for _ in range(dense_layer):
                model.add(Dense(layer_size))
                model.add(Activation('relu'))

            # adds a dense layer with sigmoid activation function to get a value between 0 and 1
            model.add(Dense(1))
            model.add(Activation('sigmoid'))

            # creates a log
            tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

            # configures the model for training using binary crossentropy and adam
            model.compile(loss='binary_crossentropy',
                          optimizer='adam',
                          metrics=['accuracy'],
                          )

            # trains the model for a given number of epochs
            model.fit(X, y,
                      batch_size=32,
                      epochs=20,
                      validation_split=0.3,
                      callbacks=[tensorboard])

# this saves the model under a certain name
# change the name between the quotation marks to change the name the model will be saved under
# change this for every model or else the new model overwrite the old one with the same name
model.save('new_model.model')
