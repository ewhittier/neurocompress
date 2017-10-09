'''Example script to generate text from Nietzsche's writings.
At least 20 epochs are required before the generated text
starts sounding coherent.
It is recommended to run this script on GPU, as recurrent
networks are quite computationally intensive.
If you try this script on new data, make sure your corpus
has at least ~100k characters. ~1M is better.
'''
from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop, Adagrad 
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import json
from keras.models import model_from_json

np.random.seed(7)

# path = get_file('nietzsche.txt', origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
# text = open(path).read().lower()
#text = "abcdefghijklmnopqrstuvwxyz" * 10

text = ''

with open('alice_small.txt', 'r') as f:
  text = f.read()

print('corpus length:', len(text))

chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 60
step = 1
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences))

print('Vectorization...')
X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1


# build the model: a single LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(256, input_shape=(maxlen, len(chars)), return_sequences = False))
#model.add(LSTM(64, return_sequences = True))
#model.add(LSTM(128))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

optimizer = RMSprop(lr = 0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    #print(len(preds))
    preds = np.asarray(preds).astype('float64')
    #print(len(preds))
    #print(type(temperature))
    preds = np.log(preds) / temperature
    
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def sample_det(preds, temperature=1.0):
    # helper function to sample an index from a probability array deterministically
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    probs = exp_preds / np.sum(exp_preds)
    return np.argmax(probs)

def predict(sentence, model, indices_char, diversity=0.1):
    x = np.zeros((1, maxlen, len(chars)))
    for t, char in enumerate(sentence):
        x[0, t, char_indices[char]] = 1.

    preds = model.predict(x, verbose=0)[0]
    next_index = sample_det(preds, diversity)
    next_char = indices_char[next_index]
    return next_char

def comparess(file1, model, indices_char):
    #this is painfully slow
    #if at all possible it should be revised so that it can mostly be run on the gpu
    #by painfully slow i mean on the order of .02 seconds per character guess. 
    #ie ~16 minutes for a 50k character file.

    f1 = open(file1, 'r').read()
    #f1 = "abcdefghijklmnopqrstuvwxyz" * 10
    data_size = len(f1)
    i = 0
    output = [0, f1[0]]
    
    data_size = len(f1)

    
    guesses_right = 0
    gss = ''
    current = f1[0]
    print("current:", current)
    while i <= data_size-2:
        #print("i:",i)
        if i < maxlen:
            gss = f1[i+1]
        else:
            #print(type(f1))
            #print(f1[i-maxlen+1:i+1])
            gss = predict(f1[(i-maxlen)+1:i+1] , model, indices_char)

        print("guess:", gss)
        #print(f1[i+1])

        if i < maxlen:
            output.append(guesses_right)
            output.append(f1[i+1])
        elif(f1[i+1] == gss):
            guesses_right += 1
        else:
            output.append(guesses_right)
            output.append(f1[i+1])
            guesses_right = 0

        current = f1[i+1]
        i += 1
        print(i)

    if output == []: output.append(data_size)
    return output

def guess(char):
    seed = [char_to_int[char]]
    x = numpy.reshape(seed, (1, len(seed), 1))
    x = x / float(vocab_size)
    prediction = model.predict(x, verbose=0)
    index = numpy.argmax(prediction)
    return int_to_char[index]

def decomparess(in_list, model):
    ls = in_list
    decomped = []
    pos = 0
    for i, entry in enumerate(ls):
        if i < maxlen*2:
            if i % 2 == 1:
                decomped.append(entry)
                pos += 1
        elif i % 2 == 0:
            for x in range(entry):
                #print(decomped[(pos - maxlen): pos])
                decomped.append(predict(decomped[(pos - maxlen):pos]  , model, indices_char))
                pos += 1
        else:
            decomped.append(entry)
            pos += 1
    return ''.join(decomped)

def train(model = model, bsize = 256, epochs = 1, iters = 1):
    for iteration in range(iters):
        print()
        print('-' * 50)
        print('Iteration', iteration)
        model.fit(X, y,
              batch_size= bsize,
              epochs=epochs)

def save(model_path = 'first(128,40)/model.json', weights_path = 'first(128,40)/weights.h5'):
    model_json = model.to_json()
    with open(model_path, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(weights_path)
    print("Saved model to disk")
    
def load(model_path = 'first(128,40)/model.json', weights_path = 'first(128,40)/weights.h5'):
    json_file = open(model_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(weights_path)

    print("Loaded model from disk")
    return loaded_model

#print(model.get_weights())

#model = load('saves/1/model.json', 'saves/1/weights.h5')

# train the model, output generated text after each iteration
'''for iteration in range(1):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(X, y,
              batch_size=128,
              epochs=10)
'''

'''with open('model.json') as f:
    jstr = json.load(f)

model = 
'''
'''
comped = comparess('alice.txt', model, indices_char)
import json
jstr =json.dumps(comped)
with open('comped.txt','w') as f:
    f.write(jstr)
print(comped)

decomped = decomparess(comped, model)

print(decomped)

with open('decomped.txt','w') as f:
    f.write(decomped)  
'''



