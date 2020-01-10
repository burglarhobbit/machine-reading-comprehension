import json

from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Activation,Dense,merge
from keras.layers import Flatten,add,multiply
from keras.layers import Bidirectional
from keras.layers.recurrent import GRU
from keras.layers.embeddings import Embedding

file_n = 'train_v1.1.json'

def build_model_synthesis():
	NUM_WORDS_Q = 50
	NUM_WORDS_P = 250
	embedding_size = 300
	hidden_layers = 150
	passage_input = Input(shape=(embedding_size,NUM_WORDS_Q,1), dtype='float32')
	question_input = Input(shape=(embedding_size,NUM_WORDS_P,1),dtype='float32')

	ht_P = Bidirectional(LSTM(hidden_layers,return_sequences=True))(passage_input)
	ht_Q = Bidirectional(LSTM(hidden_layers,return_sequences=True))(question_input)

	h1_P = Dense(embedding_size)(ht_P)
	h1_Q = Dense(embedding_size)(ht_Q)

	concat_h = keras.layers.concatenate([ht_P,ht_Q],axis=1)
	concat_w = keras.layers.concatenate([passage_input,question_input],axis=1)

	### cyclic inputs at each time steps
	dt = GRU(2*hidden_layers,return_sequences=True)
	Wa = TimeDistributed(Dense(2*hidden_layers))(dt)

	Ua = Dense(2*hidden_layers)(concat_h)

	Wa_Ua = keras.layers.add([Wa,Ua])
	Wa_Ua = Activation('tanh')

	Sj = Dense(2*hidden_layers,activation='softmax')(Wa_Ua)
	Ct = multiply(Sj,concat_h),axis=1
	###

	# read out state
	Wr = Dense(hidden_layers)(concat_w)
	Ur = Dense(hidden_layers)(Ct)
	Vr = Dense(hidden_layers)(dt)

	# maxout and softmax on W*maxout yet to be done
	

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