import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.utils import np_utils
from keras.models import model_from_json

import json
import numpy as np
import random
import sys

# fix random seed for reproducibility
numpy.random.seed(7)
    
batch_size = 1
seq_length = 1

def data_from_file(file = 'input.txt'):
    data = open(file, 'r').read() # should be simple plain text file
    chars = sorted(list(set(data)))
    data_size, vocab_size = len(data), len(chars)

    print('total chars:', vocab_size)

    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))
    return data, chars, data_size, vocab_size, char_indices, indices_char 

# define the raw dataset
#alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

def model_from_file(file = 'input.txt'):
    
    data, chars, data_size, vocab_size, char_to_int, int_to_char   = data_from_file(file)

    # create mapping of characters to integers (0-25) and the reverse
    # char_to_int = dict((c, i) for i, c in enumerate(alphabet))
    # int_to_char = dict((i, c) for i, c in enumerate(alphabet))

    # prepare the dataset of input to output pairs encoded as integers

    seq_length = 1
    dataX = []
    dataY = []
    for i in range(0, data_size - seq_length, 1):
        seq_in = data[i:i + seq_length]
        seq_out = data[i + seq_length]
        dataX.append([char_to_int[char] for char in seq_in])
        dataY.append(char_to_int[seq_out])
        print(seq_in, "->", seq_out)

    # reshape X to be [samples, time steps, features]
    X = numpy.reshape(dataX, (len(dataX), seq_length, 1))

    # normalize
    X = X / float(vocab_size)

    print("check1")
    # one hot encode the output variable
    y = np_utils.to_categorical(dataY)

    # create and fit the model
    batch_size = 1
    model = Sequential()
    model.add(LSTM(16, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
    model.add(Dense(y.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model, X, y, data, chars, data_size, vocab_size, char_to_int, int_to_char

class Model_container:

    def __init__(self, file = 'input.txt'):
        self.model, self.X, self.y, self.data, self.chars, self.data_size, self.vocab_size, self.char_to_int, self.int_to_char = model_from_file(file)
             
    def run_x_epochs(self, num, X, y, batch_size=1):
        for i in range(num):
            print(i)
            self.model.fit(self.X, self.y, epochs=1, batch_size=batch_size, verbose=2, shuffle=False)
            self.model.reset_states()

        # summarize performance of the model
        scores = self.model.evaluate(X, y, batch_size=batch_size, verbose=0)
        self.model.reset_states()
        print("Model Accuracy: %.2f%%" % (scores[1]*100))

    def predict_check(self):
        # demonstrate some model predictions
        seed = [self.char_to_int[data[0]]]
        for i in range(0, self.vocab_size-1):
            x = numpy.reshape(seed, (1, len(seed), 1))
            x = x / float(self.vocab_size)
            prediction = self.model.predict(x, verbose=0)
            index = numpy.argmax(prediction)
            print(self.int_to_char[seed[0]], "->", self.int_to_char[index])
            seed = [index]
        self.model.reset_states()

        # demonstrate a random starting point
        letter = "K"
        seed = [self.char_to_int[letter]]
        print("New start: ", letter)
        for i in range(0, 5):
            x = numpy.reshape(seed, (1, len(seed), 1))
            x = x / float(self.vocab_size)
            prediction = self.model.predict(x, verbose=0)
            index = numpy.argmax(prediction)
            print(self.int_to_char[seed[0]], "->", self.int_to_char[index])
            seed = [index]
        self.model.reset_states()

    def save(json_path = 'model.json', weights_path = 'weights.hdf5'):

        with open(json_path, 'w') as outfile:
            json.dump(self.model.to_json(), outfile)

        self.model.save_weights(weights_path)
        return


    '''def comparess(file1, file2):

        f1 = open(file1, 'r').read()
        f2 = open(file2, 'r').read()

        i = 0
        output = []

        data_size = len(f1)
        check_size = len(f2)

        if (data_size != check_size):
            print("data size doesn't match check size... continuing anyway")

        guesses_right = 0
        while i < data_size:
            
            if f1[i] == f2[i]:
                guesses_right += 1
            else:
                output.append(guesses_right)
                output.append(f2[i])
                guesses_right = 0

            if guesses_right >= data_size:
                output.append(data_size)
            
            i += 1
        return output'''

    def comparess(self, file):

        f1 = open(file, 'r').read()
        data_size = len(f1)
        i = 0
        output = []

        data_size = len(f1)

        guesses_right = 0
        current = f1[0]
        while i <= data_size-1:
            guess = self.guess(current)
            if f1[i+1] == guess:
                guesses_right += 1
            else:
                output.append(guesses_right)
                output.append(f[i+1])
                guesses_right = 0

            if guesses_right >= data_size:
                output.append(data_size)
            
            current = f[i+1]
            i += 1
        return output

    def guess(self, char):
        seed = [self.char_to_int[char]]
        x = numpy.reshape(seed, (1, len(seed), 1))
        x = x / float(self.vocab_size)
        prediction = self.model.predict(x, verbose=0)
        index = numpy.argmax(prediction)
        return self.int_to_char[index]
            
def run_x_epochs(num, X, y, batch_size=1):
    for i in range(num):
        print(i)
        model.fit(X, y, epochs=1, batch_size=batch_size, verbose=2, shuffle=False)
        model.reset_states()

    # summarize performance of the model
    scores = model.evaluate(X, y, batch_size=batch_size, verbose=0)
    model.reset_states()
    print("Model Accuracy: %.2f%%" % (scores[1]*100))

def predict_check():
    # demonstrate some model predictions
    seed = [char_to_int[data[0]]]
    for i in range(0, vocab_size-1):
        x = numpy.reshape(seed, (1, len(seed), 1))
        x = x / float(vocab_size)
        prediction = model.predict(x, verbose=0)
        index = numpy.argmax(prediction)
        print(int_to_char[seed[0]], "->", int_to_char[index])
        seed = [index]
    model.reset_states()

    # demonstrate a random starting point
    letter = "K"
    seed = [char_to_int[letter]]
    print("New start: ", letter)
    for i in range(0, 5):
        x = numpy.reshape(seed, (1, len(seed), 1))
        x = x / float(vocab_size)
        prediction = model.predict(x, verbose=0)
        index = numpy.argmax(prediction)
        print(int_to_char[seed[0]], "->", int_to_char[index])
        seed = [index]
    model.reset_states()

def save(json_path = 'model.json', weights_path = 'weights.hdf5'):

    with open(json_path, 'w') as outfile:
        json.dump(model.to_json(), outfile)

    model.save_weights(weights_path)
    return


'''def comparess(file1, file2):

    f1 = open(file1, 'r').read()
    f2 = open(file2, 'r').read()

    i = 0
    output = []

    data_size = len(f1)
    check_size = len(f2)

    if (data_size != check_size):
        print("data size doesn't match check size... continuing anyway")

    guesses_right = 0
    while i < data_size:
        
        if f1[i] == f2[i]:
            guesses_right += 1
        else:
            output.append(guesses_right)
            output.append(f2[i])
            guesses_right = 0

        if guesses_right >= data_size:
            output.append(data_size)
        
        i += 1
    return output'''

def comparess(file):

    f1 = open(file, 'r').read()
    data_size = len(f1)
    i = 0
    output = []

    data_size = len(f1)

    guesses_right = 0
    gss = ''
    current = f1[0]
    while i <= data_size-2:
        gss = guess(current)
        if f1[i+1] == guess:
            guesses_right += 1
        else:
            output.append(guesses_right)
            output.append(f1[i+1])
            guesses_right = 0

        if guesses_right >= data_size:
            output.append(data_size)
        
        current = f1[i+1]
        i += 1
    return output



def guess(char):
    seed = [char_to_int[char]]
    x = numpy.reshape(seed, (1, len(seed), 1))
    x = x / float(vocab_size)
    prediction = model.predict(x, verbose=0)
    index = numpy.argmax(prediction)
    return int_to_char[index]


#   model, X, y, data, chars, data_size, vocab_size, char_to_int, int_to_char = model_from_file()
