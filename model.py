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
	NUM_WORDS_Q = 15
	NUM_WORDS_P = 15
	embedding_size = 300
	hidden_layer = 150
	passage_input = Input(shape=(aux_inp.shape[1],1), dtype='float32')
	question_input = Input(shape=(mn_inp.shape[1],),dtype='float32')

	ht_P = Bidirectional(LSTM(hidden_layer,return_sequences=True))(passage_input)
	ht_Q = Bidirectional(LSTM(hidden_layer,return_sequences=True))(question_input)

	h1_P = Dense(embedding_size)
	h1_Q = Dense(embedding_size)

def build_model_extraction():
	"""
	model = Sequential()
	model.add(Bidirectional(GRU(10, return_sequences=True),
	                        input_shape=(5, 10)))
	model.add(Bidirectional(GRU(10)))
	model.add(Dense(5))
	model.add(Activation('softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
	return model
	"""