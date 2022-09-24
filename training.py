import random 
import json
import pickle
import numpy as np

import nltk 
from nltk.stem import WordNetLemmatizer

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, BatchNormalization

#from keras.callbacks import TensorBoard
#from keras.callbacks import ModelCheckpoint

#from keras.optimizers import gradient_descent_v2
from keras.optimizers import SGD



lemmatizer = WordNetLemmatizer()

intents = json.loads(open('intents.json').read())

words = []
classes = []
documents = []
ignore_letters = [',','.','?','!']


#############################
### Loading Training Data ###
#############################
for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))

        if intent['tag'] not in classes:
            classes.append(intent['tag'])
#print(documents)

words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))
classes = sorted(set(classes))
#print(words)

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

training = []
output_empty = [0] * len(classes)


############################
###  Words into Numbers  ###
############################
##  Using "bag of words", set indivudal words values to either 0 or 1 depending on its pattern occurance  ##
for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training)

train_x = list(training[:, 0])
train_y = list(training[:, 1])


###################################
### Creating the Neural Network ###                                                
###################################
model = Sequential()
#input layer 128
model.add(Dense(100, input_shape=(len(train_x[0]),), activation = 'relu'))
model.add(Dropout(0.5))
#hidden layer 64
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))
#output layer 
model.add(Dense(len(train_y[0]), activation = 'softmax'))

#Gradient Descent Algo
sgd = SGD(lr = 0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer = sgd, loss = 'categorical_crossentropy', metrics = ['accuracy'])       #OVERFITTED!!!!!!


hist = model.fit(np.array(train_x), np.array(train_y), epochs=100, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)

print("Training is Done")