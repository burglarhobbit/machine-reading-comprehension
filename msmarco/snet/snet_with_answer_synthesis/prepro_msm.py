import tensorflow as tf
import random
from tqdm import tqdm
import spacy
import json
from collections import Counter
import numpy as np

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


def lcs(X, Y):
	m = len(X)
	n = len(Y)
	return _lcs(X, Y, m, n)


def _lcs(X, Y, m, n):
	L = [[0 for x in range(n + 1)] for x in range(m + 1)]

	# Following steps build L[m+1][n+1] in bottom up fashion. Note
	# that L[i][j] contains length of LCS of X[0..i-1] and Y[0..j-1] 
	for i in range(m + 1):
		for j in range(n + 1):
			if i == 0 or j == 0:
				L[i][j] = 0
			elif X[i - 1] == Y[j - 1]:
				L[i][j] = L[i - 1][j - 1] + 1
			else:
				L[i][j] = max(L[i - 1][j], L[i][j - 1])

	# Following code is used to print LCS
	index = L[m][n]

	# initialized answer start and end index
	answer_start = 0
	answer_end = m
	answer_end_match = False

	# Create a character array to store the lcs string
	lcs = [""] * (index + 1)
	lcs[index] = "\0"

	# Start from the right-most-bottom-most corner and
	# one by one store characters in lcs[]
	i = m
	j = n
	while i > 0 and j > 0:

		# If current character in X[] and Y are same, then
		# current character is part of LCS
		if X[i - 1] == Y[j - 1]:
			lcs[index - 1] = X[i - 1]
			i -= 1
			j -= 1
			index -= 1
			if not answer_end_match:
				answer_end = i
				answer_end_match = True
			answer_start = i

		# If not same, then find the larger of two and
		# go in the direction of larger value
		elif L[i - 1][j] > L[i][j - 1]:
			i -= 1
		else:
			j -= 1
	# print "LCS of " + X + " and " + Y + " is " + "".join(lcs)
	# if answer_start == answer_end:
	#   answer_end += 1
	return answer_start, answer_end + 1


"""
def _lcs(X, Y, m, n):
	L = [[0 for x in range(n+1)] for x in range(m+1)]
	for i in range(m+1):
		for j in range(n+1):
			if i == 0 or j == 0:
				L[i][j] = 0
			elif X[i-1] == Y[j-1]:
				L[i][j] = L[i-1][j-1] + 1
			else:
				L[i][j] = max(L[i-1][j], L[i][j-1])
	index = L[m][n]
	answer_start = 0
	answer_end = len(X)
	answer_end_match = False
	lcs = [""] * (index+1)
	lcs[index] = "\0"   
	i = m
	j = n
	while i > 0 and j > 0:  
		if X[i-1] == Y[j-1]:
			lcs[index-1] = X[i-1]
			i-=1
			j-=1
			index-=1
			if not answer_end_match:
				answer_end = i
				answer_end_match = True
			answer_start = i
		elif L[i-1][j] > L[i][j-1]:
			i-=1
		else:
			j-=1
	return answer_start,answer_end
"""


def process_file(filename, data_type, word_counter, char_counter):
	print("Generating {} examples...".format(data_type))
	examples = []
	eval_examples = {}
	total = 0

	fh = open(filename, "r")
	line = fh.readline()
	line_limit = 300
	if data_type == "train":
		total_lines = 82326  # ms marco training data set lines
	elif data_type == "dev":
		total_lines = 10047  # ms marco dev data set lines
	elif data_type == "test":
		total_lines = 10047  # ms marco dev data set lines (for test, we use dev data set)
	line_count = 0

	do_skip_lines = False
	skip = 1330 + 789
	if do_skip_lines:
		for _ in range(skip):
			next(fh)

	# total_lines = line_limit
	# while(line):
	for i in tqdm(range(total_lines)):
		source = json.loads(line)
		answer = source['answers']
		if answer == []:
			line = fh.readline()
			continue
		elif len(answer) > 1:
			answer_text = answer[0].strip()
			if answer_text == [] or answer_text == '':
				answer_text = answer[1].strip()
				print(True)
		else:
			answer_text = answer[0].strip()
		passage_concat = ''
		# for pi, p in enumerate(article["paragraphs"]):
		for passage in source['passages']:
			passage_concat += passage['passage_text'].replace(
				"''", '" ').replace("``", '" ')
		passage_tokens = word_tokenize(passage_concat)
		passage_chars = [list(token) for token in passage_tokens]
		spans = convert_idx(passage_concat, passage_tokens)

		# word_counter increase for every qa pair. i.e. 1 since ms marco has 1 qa pair per para
		for token in passage_tokens:
			word_counter[token] += 1
			for char in token:
				char_counter[char] += 1
		ques = source['query'].replace(
			"''", '" ').replace("``", '" ')
		ques_tokens = word_tokenize(ques)
		ques_chars = [list(token) for token in ques_tokens]
		for token in ques_tokens:
			word_counter[token] += 1
			for char in token:
				char_counter[char] += 1
		y1s, y2s = [], []
		answer_texts = [answer_text]
		answer_start, answer_end = lcs(passage_concat.lower(), answer_text.lower())
		answer_span = []

		temp_span = []

		# word index for answer span
		for idx, span in enumerate(spans):
			# if not (answer_end <= span[0] or answer_start >= span[1]):
			if not (answer_end <= span[0] or answer_start >= span[1]):
				answer_span.append(idx)
			else:
				temp_span.append(span)
		try:
			y1, y2 = answer_span[0], answer_span[-1]
		except Exception as e:
			print(temp_span)
			print(answer_span, answer_start, answer_end)
			print(passage_concat)
			print(answer_text)
			continue
		y1s.append(y1)
		y2s.append(y2)
		total += 1
		example = {"passage_tokens": passage_tokens, "passage_chars": passage_chars, "ques_tokens": ques_tokens,
				   "ques_chars": ques_chars, "y1s": y1s, "y2s": y2s, "id": total}
		examples.append(example)
		eval_examples[str(total)] = {
			"passage_concat": passage_concat, "spans": spans, "answers": answer_texts, "uuid": source["query_id"]}
		line = fh.readline()
	random.shuffle(examples)
	print("{} questions in total".format(len(examples)))

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

	new_dict = {}
	for i, key in enumerate(sorted(counter, key=counter.get, reverse=True)):
		if i < 30000-5:
			new_dict[key] = counter[key]
	counter = new_dict
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
	UNK = "<UNK>"
	SOS = "<S>"
	EOS = "</S>"
	token2idx_dict = {token: idx for idx, token in enumerate(embedding_dict.keys(), 5)}
	token2idx_dict[NULL] = 0  # for padding
	token2idx_dict[OOV] = 1
	token2idx_dict[UNK] = 2
	token2idx_dict[SOS] = 3
	token2idx_dict[EOS] = 4
	embedding_dict[NULL] = [0. for _ in range(vec_size)]
	embedding_dict[OOV] = [0. for _ in range(vec_size)]
	embedding_dict[UNK] = [0. for _ in range(vec_size)]
	embedding_dict[SOS] = [0. for _ in range(vec_size)]
	embedding_dict[EOS] = [0. for _ in range(vec_size)]
	idx2emb_dict = {idx: embedding_dict[token]
					for token, idx in token2idx_dict.items()}
	emb_mat = [idx2emb_dict[idx] for idx in range(len(idx2emb_dict))]
	return emb_mat, token2idx_dict


def build_features(config, examples, data_type, out_file, word2idx_dict, is_test=False):
	para_limit = config.test_para_limit if is_test else config.para_limit
	ques_limit = config.test_ques_limit if is_test else config.ques_limit
	char_limit = config.char_limit

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
			continue

		total += 1
		passage_idxs = np.zeros([para_limit], dtype=np.int32)
		passage_char_idxs = np.zeros([para_limit, char_limit], dtype=np.int32)
		ques_idxs = np.zeros([ques_limit], dtype=np.int32)
		ques_char_idxs = np.zeros([ques_limit, char_limit], dtype=np.int32)
		y1 = np.zeros([para_limit], dtype=np.float32)
		y2 = np.zeros([para_limit], dtype=np.float32)

		def _get_word(word):
			for each in (word, word.lower(), word.capitalize(), word.upper()):
				if each in word2idx_dict:
					return word2idx_dict[each]
			return 1

		"""
		def _get_char(char):
			if char in char2idx_dict:
				return char2idx_dict[char]
			return 1
		"""
		for i, token in enumerate(example["passage_tokens"]):
			passage_idxs[i] = _get_word(token)

		for i, token in enumerate(example["ques_tokens"]):
			ques_idxs[i] = _get_word(token)

		"""
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
		"""
		start, end = example["y1s"][-1], example["y2s"][-1]
		y1[start], y2[end] = 1.0, 1.0

		record = tf.train.Example(features=tf.train.Features(feature={
			"passage_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[passage_idxs.tostring()])),
			"ques_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[ques_idxs.tostring()])),
			"passage_char_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[passage_char_idxs.tostring()])),
			"ques_char_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[ques_char_idxs.tostring()])),
			"y1": tf.train.Feature(bytes_list=tf.train.BytesList(value=[y1.tostring()])),
			"y2": tf.train.Feature(bytes_list=tf.train.BytesList(value=[y2.tostring()])),
			"id": tf.train.Feature(int64_list=tf.train.Int64List(value=[example["id"]]))
		}))
		writer.write(record.SerializeToString())
	print("Build {} / {} instances of features in total".format(total, total_))
	meta["total"] = total
	writer.close()
	return meta


def save(filename, obj, message=None):
	if message is not None:
		print("Saving {}...".format(message))
		with open(filename, "w") as fh:
			json.dump(obj, fh)


def prepro(config):
	word_counter, char_counter = Counter(), Counter()
	train_examples, train_eval = process_file(
		config.train_file, "train", word_counter, char_counter)
	dev_examples, dev_eval = process_file(
		config.dev_file, "dev", word_counter, char_counter)
	test_examples, test_eval = process_file(
		config.test_file, "test", word_counter, char_counter)
	word_emb_mat, word2idx_dict = get_embedding(
		word_counter, "word", emb_file=config.glove_file, size=config.glove_size, vec_size=config.glove_dim)
	# char_emb_mat, char2idx_dict = get_embedding(
	#	char_counter, "char", vec_size=config.char_dim)
	build_features(config, train_examples, "train",
				   config.train_record_file, word2idx_dict)
	dev_meta = build_features(config, dev_examples, "dev",
							  config.dev_record_file, word2idx_dict)
	test_meta = build_features(config, test_examples, "test",
							   config.test_record_file, word2idx_dict, is_test=True)
	save(config.word_emb_file, word_emb_mat, message="word embedding")
	# save(config.char_emb_file, char_emb_mat, message="char embedding")
	save(config.train_eval_file, train_eval, message="train eval")
	save(config.dev_eval_file, dev_eval, message="dev eval")
	save(config.test_eval_file, test_eval, message="test eval")
	save(config.dev_meta, dev_meta, message="dev meta")
	save(config.test_meta, test_meta, message="test meta")
