import json
import numpy as np
import sys
import re
from keras.preprocessing.text import Tokenizer

if (sys.version_info > (3, 0)):
	import _pickle as pickle
else:
	import cPickle as pickle

# training
file_t = 'train_v1.1.json'

# sample
file_s = 'sample.json'

GLOVE_PATH = "./glove.6B/glove.6B.300d.txt"

def tokenize(data):
	return [x.strip().lower() for i in re.split('(\W+)?', data) if i.strip()]
			
def get_all_qpa_tokens(file_n=file_s):

	question = ''
	passage = []
	answer = ''
	with open(file_n) as f:
		line = f.readline()
		#print line
		dic = json.loads(line)
		j = []
		question = dic['query']
		answer = dic['answers'][0]
		for i in dic.keys():
			#print dic[i]
			j += [i]
		for i in dic['passages']:
			#passage.append(i['passage_text'].encode('utf-8').strip())
			passage_text = i['passage_text']
			passage.append(i['passage_text'])
			"""
			if type(dic[i]) == type({}):
				for j in dic[i].keys():
					print i,":",j
			elif type(dic[i]) == type([]):
				for k in dic[i]:
					print k
			else:
				print dic[i]
				#print dic[i]
			import get_inputs as g
			g.get_all()
			"""
		question = tokenize(question)
		passage = [tokenize(i) for i in passage]
		answer = tokenize(answer)
	return question_t,passage_t,answer_t

def save_glove_300d_dic_numpy(path=GLOVE_PATH):
	import os
	embeddings_index = {}
	#embeddings_vector = np.array()
	f = open(path)
	for line in f:
		values = line.split()
		word = values[0]
		coefs = np.asarray(values[1:], dtype='float32')
		embeddings_index[word] = coefs
	with open('./glove.6B/glove.6B.300d.pickle','wb') as handle:
		#pickle.dump(embeddings_index, handle, protocol=pickle.HIGHEST_PROTOCOl)
		pickle.dump(embeddings_index, handle)

def load_embeddings():
	with open('./glove.6B/glove.6B.300d.pickle','rb') as handle:
		#pickle.dump(embeddings_index, handle, protocol=pickle.HIGHEST_PROTOCOl)
		embeddings = pickle.load(handle)
		return embedding

def get_all_embeddings():
	question_t,passage_t,answer_t = get_all_qpa_tokens()
	embedding = load_embeddings()
	

def get_all_questions():
	pass

def get_all_passages():
	pass

def get_all_answers():
	pass

def get_embeddings(data):
	embedding = load_embeddings()