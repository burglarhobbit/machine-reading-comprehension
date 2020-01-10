import tensorflow as tf
import random
from tqdm import tqdm
import spacy
import json
from collections import Counter
import numpy as np
from nltk.tokenize.moses import MosesDetokenizer
from rouge import Rouge as R
import string
import re

nlp = spacy.blank("en")


def word_tokenize(sent):
	doc = nlp(sent)
	return [token.text for token in doc]

def convert_idx(text, tokens):
	current = 0
	spans = []
	for token in tokens:
		current = text.find(token, current)
		if current < 0:
			print("Token {} cannot be found".format(token))
			raise Exception()
		spans.append((current, current + len(token)))
		current += len(token)
	return spans

# Dynamic programming implementation of LCS problem
# Returns length of LCS for X[0..m-1], Y[0..n-1] 
# Driver program
"""
X = "AGGTAB"
Y = "GXTXAYB"
lcs(X, Y)
"""

def lcs(X,Y):
	m = len(X)
	n = len(Y)
	return _lcs(X,Y,m,n)

def lcs_tokens(X,Y):
	m = len(X)
	n = len(Y)
	L = [[0 for x in range(n+1)] for x in range(m+1)]

	# Following steps build L[m+1][n+1] in bottom up fashion. Note
	# that L[i][j] contains length of LCS of X[0..i-1] and Y[0..j-1]
	ignore_tokens = [",",".","?"] 
	for i in range(m+1):
		for j in range(n+1):
			if i == 0 or j == 0:
				L[i][j] = 0
			elif X[i-1] == Y[j-1]:
				if X[i-1] in ignore_tokens:
					L[i][j] = max(L[i-1][j], L[i][j-1])
				else:
					L[i][j] = L[i-1][j-1] + 1
			else:
				L[i][j] = max(L[i-1][j], L[i][j-1])

	# initialized answer start and end index
	answer_start = answer_start_i = answer_start_j = 0
	answer_end = m-1
	answer_end_match = False
	answer_start_match = False
	
	# Start from the right-most-bottom-most corner and
	# one by one store characters in lcs[]
	index_fwd = []
	i = m
	j = n
	while i > 0 and j > 0:
		if (X[i-1] == Y[j-1]) and (X[i-1] not in ignore_tokens):
			#print(X[i-1],":",i-1)
			index_fwd.append(i-1)
			i-=1
			j-=1
			if not answer_start_match:
				answer_start_match = True
			answer_start_i = i
			answer_start_j = j
	
		# If not same, then find the larger of two and
		# go in the direction of larger value
		elif L[i-1][j] > L[i][j-1]:
			i-=1
		else:
			j-=1

	index_fwd.reverse()
	index_bwd = []
	i = answer_start_i-1
	j = answer_start_j-1
	answer_end = i
	while i < m-1 and j < n-1:
		if (X[i+1] == Y[j+1]) and (X[i+1] not in ignore_tokens):
			#print(X[i+1],":",i+1)
			index_bwd.append(i+1)
			i+=1
			j+=1
			answer_end = i
			if not answer_end_match:
				#answer_start = i
				answer_end_match = True

		# If not same, then find the larger of two and
		# go in the direction of larger value
		elif L[i+1][j] > L[i][j+1]:
			i+=1
		else:
			j+=1

	index = list(set(index_fwd).intersection(set(index_bwd)))
	index.sort()
	#print(answer_start_match, answer_end_match)
	if len(index) == 1:
		index = index * 2
	#	index[1] += 1
	return index

def _lcs(X, Y, m, n):
	L = [[0 for x in range(n+1)] for x in range(m+1)]

	# Following steps build L[m+1][n+1] in bottom up fashion. Note
	# that L[i][j] contains length of LCS of X[0..i-1] and Y[0..j-1] 
	for i in range(m+1):
		for j in range(n+1):
			if i == 0 or j == 0:
				L[i][j] = 0
			elif X[i-1] == Y[j-1]:
				L[i][j] = L[i-1][j-1] + 1
			else:
				L[i][j] = max(L[i-1][j], L[i][j-1])
	
	# Following code is used to print LCS
	#index = L[m][n]

	# initialized answer start and end index
	answer_start = 0
	answer_end = m
	answer_end_match = False

	# Create a character array to store the lcs string
	#lcs = [""] * (index+1)
	#lcs[index] = "\0"
	
	# Start from the right-most-bottom-most corner and
	# one by one store characters in lcs[]
	i = m
	j = n
	while i > 0 and j > 0:
	
		# If current character in X[] and Y are same, then
		# current character is part of LCS
		if X[i-1] == Y[j-1]:
			#lcs[index-1] = X[i-1]
			i-=1
			j-=1
			#index-=1
			if not answer_end_match:
				answer_end = i
				answer_end_match = True
			answer_start = i
	
		# If not same, then find the larger of two and
		# go in the direction of larger value
		elif L[i-1][j] > L[i][j-1]:
			i-=1
		else:
			j-=1
	#print "LCS of " + X + " and " + Y + " is " + "".join(lcs)
	#if answer_start == answer_end:
	#   answer_end += 1
	return answer_start,answer_end+1

def normalize_answer(s):

	def remove_articles(text):
		return re.sub(r'\b(a|an|the)\b', ' ', text)

	def white_space_fix(text):
		return ' '.join(text.split())

	def remove_punc(text):
		exclude = set(string.punctuation)
		return ''.join(ch for ch in text if ch not in exclude)

	def lower(text):
		return text.lower()

	return white_space_fix(remove_articles(remove_punc(lower(s))))

def rouge_l(evaluated_ngrams, reference_ngrams):
	evaluated_ngrams = set(evaluated_ngrams)
	reference_ngrams = set(reference_ngrams)
	reference_count = len(reference_ngrams)
	evaluated_count = len(evaluated_ngrams)

	# Gets the overlapping ngrams between evaluated and reference
	overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
	overlapping_count = len(overlapping_ngrams)

	# Handle edge case. This isn't mathematically correct, but it's good enough
	if evaluated_count == 0:
		precision = 0.0
	else:
		precision = overlapping_count / evaluated_count
	  
	if reference_count == 0:
		recall = 0.0 
	else:
		recall = overlapping_count / reference_count
	  
	f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))

	# return overlapping_count / reference_count
	return f1_score, precision, recall

def process_file(config, max_para_count, filename, data_type, word_counter, char_counter, is_line_limit, rouge_metric):
	detokenizer = MosesDetokenizer()
	print("Generating {} examples...".format(data_type))
	examples = []

	rouge_metric = rouge_metric # 0 = f, 1 = p, 2 = r, default = r
	rouge_l_limit = 0.7
	remove_tokens = ["'",'"','.',',','']
	eval_examples = {}
	
	fh = open(filename, "r")
	line = fh.readline()
	line_limit = config.line_limit
	if data_type == "train":
		total_lines = 82326 # ms marco training data set lines
	elif data_type == "dev":
		total_lines = 10047 # ms marco dev data set lines
	elif data_type == "test":
		total_lines = 10047 # ms marco dev data set lines (for test, we use dev data set)
	line_count = 0

	do_skip_lines = False
	skip = 1330+789
	if do_skip_lines:
		for _ in range(skip):
			next(fh)

	if is_line_limit:
		total_lines = line_limit
	#while(line):
	
	total = empty_answers = 0
	low_rouge_l = np.zeros(3,dtype=np.int32)

	# token length of concatenated passages by each query id
	concat_para_length = {}
	# para exceeding length
	max_para_length = 0
	for i in tqdm(range(total_lines)):
		source = json.loads(line)
		answer_texts = []
		answer_start = answer_end = 0
		highest_rouge_l = np.zeros(3)
		extracted_answer_text = ''
		passage_concat = ''
		passage_count = 0
		passage_pr_tokens = [] # tokens in form of 400*10 passage

		if len(source['passages'])>max_para_count:
			line = fh.readline()
			empty_answers += 1
			continue
		#print("LEN:",len(source['passages']))
		for j,passage in enumerate(source['passages']):
			passage_text = passage['passage_text'].replace(
				"''", '" ').replace("``", '" ').lower()
			passage_concat += " " + passage_text

		passage_tokens = word_tokenize(passage_concat) # tokens of passage_concat
		length = len(passage_tokens)
		if length>max_para_length:
			max_para_length = length
		answer = source['answers']
		if answer == [] or answer == ['']:
			empty_answers += 1
			line = fh.readline()
			continue
		elif len(answer)>=1:
			for answer_k,k in enumerate(answer):
				if k.strip() == "":
					continue
				answer_text = k.strip().lower()
				answer_text = answer_text[:-1] if answer_text[-1] == "." else answer_text
				answer_token = word_tokenize(answer_text)
				#index = lcs_tokens(passage_tokens, answer_token)
				#print(index)
				#####################################################################
				# individual para scoring:
				fpr_scores = (0,0,0)
				token_count = 0
				passage_pr_tokens = [] # resetting as answer for loop can run multiple times
				for l, passage in enumerate(source['passages']):
					passage_text = passage['passage_text'].replace(
						"''", '" ').replace("``", '" ').lower()
					passage_token = word_tokenize(passage_text)
					
					passage_pr_tokens += [passage_token]
					
					index = lcs_tokens(passage_token, answer_token)
					try:
						start_idx, end_idx = token_count + index[0], token_count + index[-1]
						extracted_answer = detokenizer.detokenize(passage_token[index[0]:index[-1]+1],
							return_str=True)
						detoken_ref_answer = detokenizer.detokenize(answer_token, return_str=True)
						fpr_scores = rouge_l(normalize_answer(extracted_answer), \
							normalize_answer(detoken_ref_answer))
					except Exception as e: # yes/no type questions
						pass
					if fpr_scores[rouge_metric]>highest_rouge_l[rouge_metric]:
						highest_rouge_l = fpr_scores
						answer_texts = [detoken_ref_answer]
						extracted_answer_text = extracted_answer
						answer_start, answer_end = start_idx, end_idx
					token_count += len(passage_token)
			for k in range(3):
				if highest_rouge_l[k]<rouge_l_limit:
					low_rouge_l[k] += 1
			################################################################
			if highest_rouge_l[rouge_metric]<rouge_l_limit:
				#print('\nLOW ROUGE - L\n')
				line = fh.readline()
				"""
				print(passage_concat)
				print("Question:",source['query'])
				try:
					print("Start and end index:",answer_start,",",answer_end)
					print("Passage token length:",len(passage_tokens))
					print("Extracted:",extracted_answer_text)
					print("Ground truth:",answer_texts[0])
					print("Ground truth-raw:",source['answers'])
				except Exception as e:
					print("Extracted-raw:",passage_tokens[answer_start:answer_end])
					print("Ground truth:",answer_texts)
					print("Ground truth-raw:",source['answers'])
					a = input("Pause:")
					print("\n\n")
				"""
				continue
		else:
			answer_text = answer[0].strip()
		"""
		print(passage_concat)
		print("Question:",source['query'])
		try:
			print("Start and end index:",answer_start,",",answer_end)
			print("Passage token length:",len(passage_tokens))
			print("Extracted:",extracted_answer_text)
			print("Ground truth:",answer_texts[0])
			print("Ground truth-raw:",source['answers'])
		except Exception as e:
			print("Extracted-raw:",passage_tokens[answer_start:answer_end])
			print("Ground truth:",answer_texts)
			print("Ground truth-raw:",source['answers'])
			a = input("Pause:")
		print("\n\n")
		"""
		passage_chars = [list(token) for token in passage_tokens]
		passage_pr_chars = [
			[list(token) for token in passage_tokens] for passage_tokens in passage_pr_tokens
		]
		passage_count = len(passage_pr_tokens)
		spans = convert_idx(passage_concat, passage_tokens)

		# word_counter increase for every qa pair. i.e. 1 since ms marco has 1 qa pair per para
		for token in passage_tokens:
			word_counter[token] += 1
			for char in token:
				char_counter[char] += 1
		ques = source['query'].replace(
			"''", '" ').replace("``", '" ').lower()
		ques_tokens = word_tokenize(ques)
		ques_chars = [list(token) for token in ques_tokens]
		for token in ques_tokens:
			word_counter[token] += 1
			for char in token:
				char_counter[char] += 1
		y1s, y2s = [], []
		#answer_start, answer_end = lcs(passage_concat.lower(),answer_text.lower())
		answer_span = [answer_start, answer_end]

		# word index for answer span
		for idx, span in enumerate(spans):
			#if not (answer_end <= span[0] or answer_start >= span[1]):
			if not (answer_end <= span[0] or answer_start >= span[1]):
				answer_span.append(idx)
		
		y1, y2 = answer_start, answer_end	
		y1s.append(y1)
		y2s.append(y2)
		total += 1
		concat_para_length[source["query_id"]] = len(passage_tokens)

		example = {"passage_tokens": passage_tokens, "passage_chars": passage_chars, "ques_tokens": ques_tokens,
				   "ques_chars": ques_chars, "y1s": y1s, "y2s": y2s, "id": total, "uuid": source["query_id"],
				   "passage_pr_tokens": passage_pr_tokens, "passage_pr_chars":passage_pr_chars,
				   "passage_count": passage_count}
		examples.append(example)
		eval_examples[str(total)] = {
			"passage_concat": passage_concat, "spans": spans, "answers": answer_texts, "uuid": source["query_id"],
			"ques": ques}
		line = fh.readline()
		if total%1000 == 0:
			print("{} questions in total".format(len(examples)))
			print("{} questions with empty answer".format(empty_answers))
			print("{} questions with low rouge-l answers without multipara".format(low_rouge_l))
			print("{} max-para length".format(max_para_length))
	random.shuffle(examples)
	print("{} questions in total".format(len(examples)))
	print("{} questions with empty answer".format(empty_answers))
	print("{} questions with low rouge-l answers without multipara".format(low_rouge_l))
	print("{} max-para length".format(max_para_length))
	with open(data_type+'_para_metadata.json','w') as fp:
		json.dump(concat_para_length,fp)
	"""
	# original implementation for comparision purposes
	with open(filename, "r") as fh:
		source = json.load(fh)
		for article in tqdm(source["data"]):
			for para in article["paragraphs"]:
				context = para["context"].replace(
					"''", '" ').replace("``", '" ')
				context_tokens = word_tokenize(context)
				context_chars = [list(token) for token in context_tokens]
				spans = convert_idx(context, context_tokens)

				# word_counter increase for every qa pair
				for token in context_tokens:
					word_counter[token] += len(para["qas"])
					for char in token:
						char_counter[char] += len(para["qas"])
				for qa in para["qas"]:
					total += 1
					ques = qa["question"].replace(
						"''", '" ').replace("``", '" ')
					ques_tokens = word_tokenize(ques)
					ques_chars = [list(token) for token in ques_tokens]
					for token in ques_tokens:
						word_counter[token] += 1
						for char in token:
							char_counter[char] += 1
					y1s, y2s = [], []
					answer_texts = []
					for answer in qa["answers"]:
						answer_text = answer["text"]
						answer_start = answer['answer_start']
						answer_end = answer_start + len(answer_text)
						answer_texts.append(answer_text)
						answer_span = []
						for idx, span in enumerate(spans):
							if not (answer_end <= span[0] or answer_start >= span[1]):
								answer_span.append(idx)
						y1, y2 = answer_span[0], answer_span[-1]
						y1s.append(y1)
						y2s.append(y2)
					example = {"context_tokens": context_tokens, "context_chars": context_chars, "ques_tokens": ques_tokens,
							   "ques_chars": ques_chars, "y1s": y1s, "y2s": y2s, "id": total}
					examples.append(example)
					eval_examples[str(total)] = {
						"context": context, "spans": spans, "answers": answer_texts, "uuid": qa["id"]}
		random.shuffle(examples)
		print("{} questions in total".format(len(examples)))
	"""
	return examples, eval_examples

def get_embedding(counter, data_type, limit=-1, emb_file=None, size=None, vec_size=None):
	print("Generating {} embedding...".format(data_type))
	embedding_dict = {}
	filtered_elements = [k for k, v in counter.items() if v > limit]
	if emb_file is not None:
		assert size is not None
		assert vec_size is not None
		with open(emb_file, "r", encoding="utf-8") as fh:
			for line in tqdm(fh, total=size):
				array = line.split()
				word = "".join(array[0:-vec_size])
				vector = list(map(float, array[-vec_size:]))
				if word in counter and counter[word] > limit:
					embedding_dict[word] = vector
		print("{} / {} tokens have corresponding embedding vector".format(
			len(embedding_dict), len(filtered_elements)))
	else:
		assert vec_size is not None
		for token in filtered_elements:
			embedding_dict[token] = [0. for _ in range(vec_size)]
		print("{} tokens have corresponding embedding vector".format(
			len(filtered_elements)))
	NULL = "--NULL--"
	OOV = "--OOV--"
	token2idx_dict = {token: idx for idx,
					  token in enumerate(embedding_dict.keys(), 2)}
	token2idx_dict[NULL] = 0
	token2idx_dict[OOV] = 1
	embedding_dict[NULL] = [0. for _ in range(vec_size)]
	embedding_dict[OOV] = [0. for _ in range(vec_size)]
	idx2emb_dict = {idx: embedding_dict[token]
					for token, idx in token2idx_dict.items()}
	emb_mat = [idx2emb_dict[idx] for idx in range(len(idx2emb_dict))]
	return emb_mat, token2idx_dict

def build_features(config, examples, data_type, out_file, word2idx_dict, char2idx_dict, is_test=False):

	para_limit = config.test_para_limit if is_test else config.para_limit
	ques_limit = config.test_ques_limit if is_test else config.ques_limit
	char_limit = config.char_limit
	single_para_limit = spl = config.single_para_limit
	max_para = config.max_para

	def filter_func(example, is_test=False):
		return len(example["passage_tokens"]) > para_limit or len(example["ques_tokens"]) > ques_limit

	print("Processing {} examples...".format(data_type))
	writer = tf.python_io.TFRecordWriter(out_file)
	total = 0
	total_ = 0
	meta = {}
	for example in tqdm(examples):
		total_ += 1

		if filter_func(example, is_test):
			print("Filtered")
			continue

		total += 1
		passage_idxs = np.zeros([para_limit], dtype=np.int32)
		passage_char_idxs = np.zeros([para_limit, char_limit], dtype=np.int32)
		
		passage_pr_idxs = np.zeros([spl*max_para], dtype=np.int32)
		passage_pr_char_idxs = np.zeros([spl*max_para, char_limit], dtype=np.int32)
		
		ques_idxs = np.zeros([ques_limit], dtype=np.int32)
		ques_char_idxs = np.zeros([ques_limit, char_limit], dtype=np.int32)
		y1 = np.zeros([para_limit], dtype=np.float32)
		y2 = np.zeros([para_limit], dtype=np.float32)

		def _get_word(word):
			for each in (word, word.lower(), word.capitalize(), word.upper()):
				if each in word2idx_dict:
					return word2idx_dict[each]
			return 1

		def _get_char(char):
			if char in char2idx_dict:
				return char2idx_dict[char]
			return 1

		for i, token in enumerate(example["passage_tokens"]):
			passage_idxs[i] = _get_word(token)

		# for snet
		for i, paragraph in enumerate(example["passage_pr_tokens"]):
			for j, token in enumerate(paragraph):
				try:
					passage_pr_idxs[i*spl+j] = _get_word(token)
				except Exception as e:
					#print(i,j)
					pass
		
		# for snet
		for i, paragraph in enumerate(example["passage_pr_chars"]):
			for j, token in enumerate(paragraph):
				for k, char in enumerate(token):
					if k == char_limit:
						break
					passage_pr_char_idxs[i*spl+j, k] = _get_char(char)

		for i, token in enumerate(example["ques_tokens"]):
			ques_idxs[i] = _get_word(token)

		for i, token in enumerate(example["passage_chars"]):
			for j, char in enumerate(token):
				if j == char_limit:
					break
				passage_char_idxs[i, j] = _get_char(char)

		for i, token in enumerate(example["ques_chars"]):
			for j, char in enumerate(token):
				if j == char_limit:
					break
				ques_char_idxs[i, j] = _get_char(char)

		start, end = example["y1s"][-1], example["y2s"][-1]

		y1[start], y2[end] = 1.0, 1.0
		if total%config.checkpoint==0:
			print("Processed {} examples...".format(total))
		record = tf.train.Example(features=tf.train.Features(feature={
								  "passage_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[passage_idxs.tostring()])),
								  "ques_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[ques_idxs.tostring()])),
								  "passage_char_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[passage_char_idxs.tostring()])),
								  "ques_char_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[ques_char_idxs.tostring()])),
								  "passage_pr_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[passage_pr_idxs.tostring()])),
								  "passage_pr_char_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[passage_pr_char_idxs.tostring()])),
								  "y1": tf.train.Feature(bytes_list=tf.train.BytesList(value=[y1.tostring()])),
								  "y2": tf.train.Feature(bytes_list=tf.train.BytesList(value=[y2.tostring()])),
								  "id": tf.train.Feature(int64_list=tf.train.Int64List(value=[example["id"]])),
								  "passage_count": tf.train.Feature(int64_list=tf.train.Int64List(value=[example["passage_count"]])),
								  }))
		writer.write(record.SerializeToString())
	print("Build {} / {} instances of features in total".format(total, total_))
	print("Processed {} examples...".format(total))
	meta["total"] = total
	writer.close()
	return meta


def save(filename, obj, message=None):
	if message is not None:
		print("Saving {}...".format(message))
		with open(filename, "w") as fh:
			json.dump(obj, fh)


def prepro_(config):
	word_counter, char_counter = Counter(), Counter()
	train_examples, train_eval = process_file(config,
		config.max_para, config.train_file, "train", word_counter, char_counter, 
		config.line_limit_prepro, config.rouge_metric)
	dev_examples, dev_eval = process_file(config,
		config.max_para, config.dev_file, "dev", word_counter, char_counter, 
		config.line_limit_prepro, config.rouge_metric)
	test_examples, test_eval = process_file(config,
		config.max_para, config.test_file, "test", word_counter, char_counter,
		config.line_limit_prepro, config.rouge_metric)
	word_emb_mat, word2idx_dict = get_embedding(
		word_counter, "word", emb_file=config.glove_file, size=config.glove_size, vec_size=config.glove_dim)
	char_emb_mat, char2idx_dict = get_embedding(
		char_counter, "char", vec_size=config.char_dim)
	train_meta = build_features(config, train_examples, "train",
				   config.train_record_file, word2idx_dict, char2idx_dict)
	dev_meta = build_features(config, dev_examples, "dev",
							  config.dev_record_file, word2idx_dict, char2idx_dict)
	test_meta = build_features(config, test_examples, "test",
							   config.test_record_file, word2idx_dict, char2idx_dict, is_test=True)
	save(config.word_emb_file, word_emb_mat, message="word embedding")
	save(config.char_emb_file, char_emb_mat, message="char embedding")
	save(config.train_eval_file, train_eval, message="train eval")
	save(config.dev_eval_file, dev_eval, message="dev eval")
	save(config.test_eval_file, test_eval, message="test eval")
	save(config.train_meta, train_meta, message="train meta")
	save(config.dev_meta, dev_meta, message="dev meta")
	save(config.test_meta, test_meta, message="test meta")