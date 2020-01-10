import json

from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Activation,Dense
from keras.layers import Flatten
from keras.layers import Bidirectional
from keras.layers.recurrent import GRU
from keras.layers.embeddings import Embedding

from get_inputs import get_all_inputs
from model import get_model

file_n = 'train_v1.1.json'

def train():
	ques, passage, ans = get_all_inputs()