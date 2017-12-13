import json

from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Activation,Dense
from keras.layers import Flatten
from keras.layers import Bidirectional
from keras.layers.recurrent import GRU
from keras.layers.embeddings import Embedding

file_n = 'train_v1.1.json'

def build_model_synthesis():

	
def build_model_extraction():
	model = Sequential()
	model.add(Bidirectional(GRU(10, return_sequences=True),
	                        input_shape=(5, 10)))
	model.add(Bidirectional(GRU(10)))
	model.add(Dense(5))
	model.add(Activation('softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
	return model