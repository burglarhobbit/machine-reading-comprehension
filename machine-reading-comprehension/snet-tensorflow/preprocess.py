# coding=utf-8
import os
import re
from collections import Counter
import json
import numpy as np
import math
from random import randint

def sublist_exists(sl, l):
	n = len(sl)
	return any((sl == l[i:i+n]) for i in range(len(l)-n+1))

def sublist_idx(sl, l):
	sll=len(sl)
	for ind in (i for i,e in enumerate(l) if e==sl[0]):
		if l[ind:ind+sll]==sl:
			return ind,ind+sll

def tokenize(data):	
	return [i.strip().lower() for i in re.split('(\W+)?', data) if i.strip()]

class DataProcessor:
	def __init__(self, data_type, opts):
		self.data_type = data_type
		self.opts = opts
		data_path = os.path.join('Data', "data_{}.json".format(data_type))
		shared_path = os.path.join('Data', "shared_{}.json".format(data_type))
		idx_path = os.path.join('Data', "idx_table.json")
		self.data = self.load_data(data_path)
		self.shared = self.load_data(shared_path)
		self.idx_table = self.load_data(idx_path)

		# paragraph length filter: (train only)
		if self.data_type == 'train':
			self.data = [sample for sample in self.data if sample['answer'][0][-1] < self.opts['p_length']]
		self.num_samples = self.get_data_size()
		print("Loaded {} examples from {}".format(self.num_samples, data_type))

	def load_data(self, path):
		with open(path, 'r') as fh:
			data = json.load(fh)
		return data
	

	def get_data_size(self):
		return len(self.data)

	def get_training_batch(self, batch_no):
		opts = self.opts
		si = (batch_no * opts['batch_size'])
		ei = min(self.num_samples, si + opts['batch_size'])
		n = ei - si

		tensor_dict = {}
		paragraph = np.zeros((n, opts['p_length'], opts['word_emb_dim']))
		question = np.zeros((n, opts['q_length'], opts['word_emb_dim']))
		paragraph_c = np.zeros((n, opts['p_length'], opts['char_max_length']))
		question_c = np.zeros((n, opts['q_length'], opts['char_max_length']))
		answer_si = np.zeros( (n, opts['p_length']) )
		answer_ei = np.zeros( (n, opts['p_length']) )
		idxs= [] 

		count = 0
		for i in range(si, ei):
			idxs.append(i)
			sample = self.data[i]
			aipi = sample['aipi']
			p = self.shared['passages'][aipi[0]][aipi[1]]
			q = sample['question']

			for j in range(len(p)):
				if j >= opts['p_length']:
					break
				try:
					paragraph[count][j][:opts['word_emb_dim']] = self.shared['glove'+opts['glove']][p[j]]
				except KeyError:
					pass
				for k, char in enumerate(p[j]):
					paragraph_c[count][j][k] = self.idx_table['char2idx'][char]
			
			for j in range(len(q)):
				if j >= opts['q_length']:
					break
				try:
					question[count][j] = self.shared['glove'+opts['glove']][q[j]]
				except KeyError:
					pass
				for k, char in enumerate(q[j]):
					question_c[count][j][k] = self.idx_table['char2idx'][char]
			
			si, ei = sample['answer'][0][0], sample['answer'][0][-1]
			answer_si[count][si] = 1.0
			answer_ei[count][ei] = 1.0
			
			count += 1
		
		tensor_dict['paragraph'] = paragraph
		tensor_dict['question'] = question
		tensor_dict['paragraph_c'] = paragraph_c
		tensor_dict['question_c'] = question_c
		tensor_dict['answer_si'] = answer_si
		tensor_dict['answer_ei'] = answer_ei
		return tensor_dict, idxs
	
	def get_testing_batch(self, batch_no):
		opts = self.opts
		si = (batch_no * opts['batch_size'])
		ei = min(self.num_samples, si + opts['batch_size'])
		n = ei - si

		paragraph = np.zeros((opts['batch_size'], opts['p_length'], opts['word_emb_dim']))
		question = np.zeros((opts['batch_size'], opts['q_length'], opts['word_emb_dim']))
		paragraph_c = np.zeros((opts['batch_size'], opts['p_length'], opts['char_max_length']))
		question_c = np.zeros((opts['batch_size'], opts['q_length'], opts['char_max_length']))
		context = [None for _ in range(n)]
		context_original = [None for _ in range(n)]
		answer_si = [None for _ in range(n)]
		answer_ei = [None for _ in range(n)]
		ID = [None for _ in range(n)]
		
		count = 0
		for i in range(si, ei):
			sample = self.data[i]
			aipi = sample['aipi']
			p = self.shared['passages'][aipi[0]][aipi[1]]
			p_o = self.shared['passages_original'][aipi[0]][aipi[1]]
			q = sample['question']
			
			context[count] = p
			context_original[count] = p_o
			for j in range(len(p)):
				if j >= opts['p_length']:
					break
				try:
					paragraph[count][j][:opts['word_emb_dim']] = self.shared['glove'+opts['glove']][p[j]]
					for k, char in enumerate(p[j]):
						paragraph_c[count][j][k] = self.idx_table['char2idx'][char]
				except KeyError:
					#print('{} not in GloVe'.format(p[j]))
					pass
			
			for j in range(len(q)):
				if j >= opts['q_length']:
					break
				try:
					question[count][j] = self.shared['glove'+opts['glove']][q[j]]
					for k, char in enumerate(q[j]):
						question_c[count][j][k] = self.idx_table['char2idx'][char]
				except KeyError:
					pass
					#print('{} not in GloVe'.format(triplet['question'][j].lower()))
			
			answer_si[count] = [ans[0]  for ans in sample['answer']]
			answer_ei[count] = [ans[-1] for ans in sample['answer']]
			ID[count] = sample['id']
			count += 1
		
		return context, context_original, paragraph, question, paragraph_c, question_c, answer_si, answer_ei, ID, n

def get_word2vec(glove_path, word_counter):
	word2vec_dict = {}
	with open(glove_path, 'r', encoding='utf-8') as fh:
		for line in fh:
			array = line.lstrip().rstrip().split(" ")
			word = array[0]
			vector = list(map(float, array[1:]))
			if word in word_counter:
				word2vec_dict[word] = vector
			if word.capitalize() in word_counter:
				word2vec_dict[word.capitalize()] = vector
			if word.lower() in word_counter:
				word2vec_dict[word.lower()] = vector
			if word.upper() in word_counter:
				word2vec_dict[word.upper()] = vector
	print("{}/{} of word vocab have corresponding vectors in {}".format(len(word2vec_dict), len(word_counter), glove_path))
	return word2vec_dict

def get_char_vocab(word_counter):
	char2idx = {' ':0}
	idx2char = [' ']
	max_word_length = 0
	word_count = [0 for _ in range(107)] #37 original, 107 biggest word length

	for word in word_counter:
		print(word)
		word_count[len(word)-1] +=1
		max_word_length = max(max_word_length, len(word))
		for char in word:
			if not char in char2idx:
				idx2char.append(char)
				char2idx[char] = len(idx2char) - 1
	print('max word length:',max_word_length)
	print(len(char2idx),'chars read')
	print(word_count)

	return char2idx, idx2char

def read_local_word2vec():
	local_w2v_dir = os.join.path('Data', 'local_w2v', 'local_w2v.json')
	local_w2v = json.load(open(local_w2v_dir, 'r'))
	return local_w2v_dir	


"""
from preprocess import generate_seq
generate_seq('train')
"""
def generate_seq(data_type):
	import nltk
	from nltk.tokenize import word_tokenize, sent_tokenize

	fpr = open(os.path.join('Data', data_type+'_v1.1.json'), 'r')
	#source_data = json.load(fpr)
	
	data = []
	articles = []
	articles_sent = []
	articles_original = []
	articles_original_sent = []
	word_counter = Counter()

	fpw = open(os.path.join('Data','data_'+data_type+".json"), 'w')
	#for _ in range(32000):
	#	next(fpr)
	line = fpr.readline()
	ai = 0
	pi = 0
	#for ai, article in enumerate(source_data["data"]):
	while(line and ai<500):
		if ai%20 == 0:
			print('processing article',ai)
		json_line = json.loads(line)
		passages = []
		passages_sent = []
		passages_original = []
		passages_original_sent = []

		answer = json_line['answers']
		#answer_1 = ''
		if answer == []:
			line = fpr.readline()
			continue
		elif len(answer)>1:
			answer_1 = answer[0].strip()
			if answer_1 == [] or answer_1 == '':
				answer_1 = answer[1].strip()
				print(True)
		else:
			answer_1 = answer[0].strip()
		passage_concat = ''
		#for pi, p in enumerate(article["paragraphs"]):
		for passage in json_line['passages']:
			passage_concat += passage['passage_text']
			
		#context = p["context"]
		#context = context.replace("''", '" ')
		#context = context.replace("``", '" ')

		passage = word_tokenize(passage_concat)
		passage_sent = sent_tokenize(passage_concat)
		passage_sent = [word_tokenize(sent) for sent in passage_sent]
		passages.append(passage) # word level paragraph
		passages_sent.append(passage_sent) # sentence_word level paragraph
		passages_original.append(passage_concat) # original paragraph
		passages_original_sent.append(passage_sent) # sentence_tokenized original paragraph
		for w in passage:
			word_counter[w] += 1

		#for qa in p["qas"]:
		question = word_tokenize(json_line["query"])
		answers = []
		answers_sent = []
		for w in question:
			word_counter[w] += 1

		#for a in answer:
		#answer_start = int(a['answer_start'])
		
		# will have to update with some kind of span prediction using rouge-L
		answer_start = randint(0,int(len(passage_concat)*8/10))

		#add '.' here, just because NLTK is not good enough in some cases
		answer_words = word_tokenize(answer_1 + '.')
		if answer_words[-1] == '.':
			answer_words = answer_words[:-1]
		else:
			answer_words = word_tokenize(answer_1)

		#word level
		prev_context_words = word_tokenize( passage_concat[:answer_start] )
		left_context_words = word_tokenize( passage_concat[answer_start:] )
		pos_list = []
		for i in range(len(answer_words)):
			if i < len(left_context_words):
				pos_list.append(len(prev_context_words) + i)
		#assert(len(pos_list) > 0)
		if(len(pos_list) == 0):
			print(answer_words)
			print(answer)
			print(ab)
			print(question)
			assert(False)

		# sent level
		# [sent_idx, word_idx]
		for idx, sent in enumerate(passage_sent):
			if sublist_exists(answer_words, sent):
				sent_idx = idx
				try:
					si,ei = sublist_idx(answer_words, sent)
				except:
					print(answer)
					print(answer_words)
					print(sent)
					exit()
				pos_list_sent = [[idx, i] for i in range(si, ei)]
				break
			else:
				pos_list_sent = []
		
		answers.append(pos_list)
		answers_sent.append(pos_list_sent)

		sample = {'aipi': [ai, pi],
				  'question': question,
				  'answer': answers,
				  'answer_sent': answers_sent, 
				  'id': str(json_line['query_id']), 
				  }
		data.append(sample)
		articles.append(passages)
		articles_sent.append(passage_sent)
		articles_original.append(passages_original)
		articles_original_sent.append(passages_original_sent)
		ai += 1
		line = fpr.readline()
	w2v_100 = get_word2vec('./Data/glove.6B.50d.txt', word_counter)
	#w2v_300 = get_word2vec('./Data/glove.840B.300d.txt', word_counter)
	char2idx, idx2char = get_char_vocab(word_counter)

	print(len(data))
	print(len(articles), len(articles_sent))
	shared = {'passages': articles,
			  'passages_sent': articles_sent,
			  'passages_original': articles_original,
			  'passages_original_sent': articles_original_sent,
			  'glove100': w2v_100,
			  #'glove300': w2v_300,
			  }
	print('Saving...')
	with open(os.path.join('Data','data_'+data_type+".json"), 'w') as f:
		json.dump(data, f)
	with open(os.path.join('Data','shared_'+data_type+".json"), 'w') as f:
		json.dump(shared, f)

	if data_type == 'train':
		char2idx, idx2char = get_char_vocab(word_counter)
		idx_table = {'char2idx': char2idx,
					 'idx2char': idx2char,
					 }
		with open(os.path.join('Data','idx_table.json'), 'w') as f:
			json.dump(idx_table, f)

	print('MS Marco '+data_type+' preprossing finished!')
	"""
from preprocess import generate_seq as g
g('train')
	"""
def read_data(data_type, opts):
	return DataProcessor(data_type, opts)

def run():
	import argparse
	
	parser = argparse.ArgumentParser()
	parser.add_argument('--gen_seq', type=bool, default=False, help='original data to seq')
	args = parser.parse_args()
	
	if args.gen_seq:
		print('Generating Sequences...')
		generate_seq('train')
		generate_seq('dev')

if __name__ == "__main__":
	#run()
	generate_seq('train')