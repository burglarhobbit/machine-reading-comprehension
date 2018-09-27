""" Official evaluation script for v1.1 of the SQuAD dataset. """
from __future__ import print_function
from collections import Counter
import string
import re
import argparse
import json
import sys


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


def f1_score(prediction, ground_truth):
	prediction_tokens = normalize_answer(prediction).split()
	ground_truth_tokens = normalize_answer(ground_truth).split()
	common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
	num_same = sum(common.values())
	if num_same == 0:
		return 0
	precision = 1.0 * num_same / len(prediction_tokens)
	recall = 1.0 * num_same / len(ground_truth_tokens)
	f1 = (2 * precision * recall) / (precision + recall)
	return f1


def exact_match_score(prediction, ground_truth):
	return (normalize_answer(prediction) == normalize_answer(ground_truth))

def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
	scores_for_ground_truths = []
	for ground_truth in ground_truths:
		score = metric_fn(prediction, ground_truth)
		scores_for_ground_truths.append(score)
	return max(scores_for_ground_truths)

def rouge_l(rouge_obj, prediction, ground_truth):

	prediction_tokens = normalize_answer(prediction)
	ground_truth_tokens = normalize_answer(ground_truth)
	scores = rouge_obj.get_scores(prediction_tokens, ground_truth_tokens)
	rouge_l_ = scores[0]['rouge-l']['r']
	#print(prediction_tokens)
	#print(ground_truth_tokens)
	return rouge_l_

def evaluate(eval_file, answer_dict):
	f1 = rouge_l_= exact_match = total =  0
	from rouge import Rouge
	
	rouge = Rouge()
	#for key in answer_dict.items()

	## converting eval_file keys to format of answer_dict keys format
	# i.e (remapped_answer_dict format): read utils->convert_tokens and main.py->test last few lines
	remapped_eval_file = {}

	for key, value in eval_file.items():
		uuid = eval_file[key]["uuid"]
		#print(type(uuid))
		remapped_eval_file[str(uuid)] = eval_file[key]["answers"][0]
	
	a = remapped_eval_file.keys()
	b = []
	for i in answer_dict.keys():
		b.append(str(i))
	#print(len(a))
	#print(len(b))
	print(len(list(set(a).intersection(b))))
	for key, value in answer_dict.items():
		total += 1
		ground_truths = remapped_eval_file[str(key)]
		prediction = value
		exact_match += metric_max_over_ground_truths(
			exact_match_score, prediction, ground_truths)
		f1 += metric_max_over_ground_truths(f1_score,
											prediction, ground_truths)
		rouge_l_ += rouge_l(rouge, prediction, ground_truths)
		#print(key)
	exact_match = 100.0 * exact_match / total
	f1 = 100.0 * f1 / total
	rouge_l_ = 100.0 * rouge_l_ / total

	return {'exact_match': exact_match, 'f1': f1, 'rouge-l': rouge_l_}

if __name__ == '__main__':
	expected_version = '1.1'
	parser = argparse.ArgumentParser(
		description='Evaluation for MS-MARCO ' + expected_version)
	parser.add_argument('dataset_file', help='Dataset file')
	parser.add_argument('prediction_file', help='Prediction File')
	args = parser.parse_args()
	with open(args.dataset_file) as dataset_file:
		dataset = json.load(dataset_file)
		#if (dataset_json['version'] != expected_version):
		#   print('Evaluation expects v-' + expected_version +
		#         ', but got dataset with v-' + dataset_json['version'],
		#         file=sys.stderr)
		#dataset = dataset_json['data']
	with open(args.prediction_file) as prediction_file:
		predictions = json.load(prediction_file)
	print(json.dumps(evaluate(dataset, predictions)))
