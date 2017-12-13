import json

# training
file_t = 'train_v1.1.json'

# sample
file_s = 'sample.json'


def get_all(file_n=file_s):

	with open(file_n) as f:
		line = f.readline()
		print line
		dic = json.loads(line)
		for i in dic.keys():
			print i
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
			"""

def save_glove_300d_dic_numpy(path="./glove.6B/glove.6B.300d.txt"):
	import numpy as np
	import cPickle as pickle
	embeddings_index = {}
	#embeddings_vector = np.array()
	f = open(path)
	for line in f:
		values = line.split()
		word = values[0]
		coefs = np.asarray(values[1:], dtype='float32')
		embeddings_index[word] = coefs
	with open('glove.6B.300d.pickle','wb') as handle:
		#pickle.dump(embeddings_index, handle, protocol=pickle.HIGHEST_PROTOCOl)
		pickle.dump(embeddings_index, handle)

def get_all_embeddings():
	pass

def get_all_questions():
	pass

def get_all_passages():
	pass

def get_all_answers():
	pass

def get_embeddings():
	pass