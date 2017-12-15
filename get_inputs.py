import json
import numpy as np
import sys
import re
from keras.preprocessing.text import Tokenizer

if (sys.version_info > (3, 0)):
	import _pickle as pickle
	sys_type = 3
else:
	import cPickle as pickle
	sys_type = 2
# training
file_t = 'train_v1.1.json'

# sample
file_s = 'sample.json'

GLOVE_PATH = "./glove.6B/glove.6B.300d.txt"

def tokenize(data):
	
	return [i.strip().lower() for i in re.split('(\W+)?', data) if i.strip()]
			
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
	return question,passage,answer

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
	if sys_type == 3:
		glove_pickle = './glove.6B/glove.6B.300d.pickle3'
	elif sys_type == 2:
		glove_pickle = './glove.6B/glove.6B.300d.pickle2'
	
	with open(glove_pickle,'wb') as handle:
		#pickle.dump(embeddings_index, handle, protocol=pickle.HIGHEST_PROTOCOl)
		pickle.dump(embeddings_index, handle)

def load_embeddings():
	if sys_type == 3:
		glove_pickle = './glove.6B/glove.6B.300d.pickle3'
	elif sys_type == 2:
		glove_pickle = './glove.6B/glove.6B.300d.pickle2'
	with open(glove_pickle,'rb') as handle:
		#pickle.dump(embeddings_index, handle, protocol=pickle.HIGHEST_PROTOCOl)
		embeddings = pickle.load(handle)
		return embeddings

def get_all_embeddings():
	question_t,passage_t,answer_t = get_all_qpa_tokens()
	em = load_embeddings()
	zeros = [0]*300	
	
	question_em = em.get(question_t[0],np.array(zeros)).reshape(300,1)
	
	passage_em = em.get(passage_t[0][0],np.array(zeros)).reshape(300,1)
	
	for i in passage_t[0][1:]:
		embedding = em.get(passage_t[0][0],np.array(zeros)).reshape(300,1)
		passage_em = np.concatenate((passage_em,embedding),axis=1)
	
	answer_em = em.get(answer_t[0],np.array(zeros)).reshape(300,1)

	for i in question_t[1:]:
		embedding = em.get(i,np.array(zeros)).reshape(300,1)
		question_em = np.concatenate((question_em,embedding),axis=1)
	for j in passage_t[1:]:
		for i in j:
			embedding = em.get(i,np.array(zeros)).reshape(300,1)
			passage_em = np.concatenate((passage_em,embedding),axis=1)
	for i in answer_t[1:]:
		embedding = em.get(i,np.array(zeros)).reshape(300,1)
		answer_em = np.concatenate((answer_em,embedding),axis=1)
	return question_em,passage_em,answer_em
def get_all_questions():
	pass

def get_all_passages():
	pass

def get_all_answers():
	pass

def get_embeddings(data):
	embedding = load_embeddings()