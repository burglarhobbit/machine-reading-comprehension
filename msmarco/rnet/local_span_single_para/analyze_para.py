import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

flags = tf.flags

ome = os.path.expanduser("~")
hdd2 = "/media/hdd2/snetP_data"

if os.path.isdir(hdd2):
	path = hdd2
else:
	path = home

train_file = os.path.join(path, "data", "msmarco", "train_v1.1.json")
dev_file = os.path.join(path, "data", "msmarco", "dev_v1.1.json")
test_file = os.path.join(path, "data", "msmarco", "dev_v1.1.json")
glove_file = os.path.join(path, "data", "glove", "glove.840B.300d.txt")

#train_file = os.path.join(hdd2, "snetP_data", "data", "msmarco", "train_v1.1.json")
#dev_file = os.path.join(hdd2, "snetP_data", "data", "msmarco", "dev_v1.1.json")
#test_file = os.path.join(hdd2, "snetP_data", "data", "msmarco", "test_public_v1.1.json")
#glove_file = os.path.join(hdd2, "snetP_data", "data", "glove", "glove.840B.300d.txt")
#target_dir = os.path.join(hdd2, "snetP_data", "snet_data")

#target_dir = "data"
target_dir = os.path.join(path, "preprocess", "rnet", "msmarco", "local_span")
log_dir = os.path.join(path, "rnet", "msmarco", "local_span", "log", "event")
save_dir = os.path.join(path, "rnet", "msmarco", "local_span", "log", "model")
answer_dir = os.path.join(path, "rnet", "msmarco", "local_span", "log", "answer")

train_record_file = os.path.join(target_dir, "train.tfrecords")
dev_record_file = os.path.join(target_dir, "dev.tfrecords")
test_record_file = os.path.join(target_dir, "test.tfrecords")
word_emb_file = os.path.join(target_dir, "word_emb.json")
char_emb_file = os.path.join(target_dir, "char_emb.json")
train_eval = os.path.join(target_dir, "train_eval.json")
dev_eval = os.path.join(target_dir, "dev_eval.json")
test_eval = os.path.join(target_dir, "test_eval.json")
dev_meta = os.path.join(target_dir, "dev_meta.json")
test_meta = os.path.join(target_dir, "test_meta.json")
answer_file = os.path.join(answer_dir, "answer.json")

if not os.path.exists(target_dir):
	os.makedirs(target_dir)
if not os.path.exists(log_dir):
	os.makedirs(log_dir)
if not os.path.exists(save_dir):
	os.makedirs(save_dir)
if not os.path.exists(answer_dir):
	os.makedirs(answer_dir)

flags.DEFINE_string("mode", "train", "Running mode train/debug/test")

flags.DEFINE_string("target_dir", target_dir, "Target directory for out data")
flags.DEFINE_string("log_dir", log_dir, "Directory for tf event")
flags.DEFINE_string("save_dir", save_dir, "Directory for saving model")
flags.DEFINE_string("train_file", train_file, "Train source file")
flags.DEFINE_string("dev_file", dev_file, "Dev source file")
flags.DEFINE_string("test_file", test_file, "Test source file")
flags.DEFINE_string("glove_file", glove_file, "Glove source file")

flags.DEFINE_string("train_record_file", train_record_file,
					"Out file for train data")
flags.DEFINE_string("dev_record_file", dev_record_file,
					"Out file for dev data")
flags.DEFINE_string("test_record_file", test_record_file,
					"Out file for test data")
flags.DEFINE_string("word_emb_file", word_emb_file,
					"Out file for word embedding")
flags.DEFINE_string("char_emb_file", char_emb_file,
					"Out file for char embedding")
flags.DEFINE_string("train_eval_file", train_eval, "Out file for train eval")
flags.DEFINE_string("dev_eval_file", dev_eval, "Out file for dev eval")
flags.DEFINE_string("test_eval_file", test_eval, "Out file for test eval")
flags.DEFINE_string("dev_meta", dev_meta, "Out file for dev meta")
flags.DEFINE_string("test_meta", test_meta, "Out file for test meta")
flags.DEFINE_string("answer_file", answer_file, "Out file for answer")

flags.DEFINE_string("gpu_id", "2", "gpu id to use for training")

flags.DEFINE_integer("glove_size", int(2.2e6), "Corpus size for Glove")
flags.DEFINE_integer("glove_dim", 300, "Embedding dimension for Glove")
flags.DEFINE_integer("char_dim", 8, "Embedding dimension for char")
flags.DEFINE_integer("para_limit", 1500, "Limit length for paragraph")
flags.DEFINE_integer("max_para", 10, "Limit length for paragraph")
flags.DEFINE_integer("ques_limit", 50, "Limit length for question")
flags.DEFINE_integer("test_para_limit", 1000,
					 "Limit length for paragraph in test file")
flags.DEFINE_integer("test_ques_limit", 100,
					 "Limit length for question in test file")
flags.DEFINE_integer("char_limit", 16, "Limit length for character")
flags.DEFINE_integer("word_count_limit", -1, "Min count for word")
flags.DEFINE_integer("char_count_limit", -1, "Min count for char")

flags.DEFINE_integer("capacity", 15000, "Batch size of dataset shuffle")
flags.DEFINE_integer("num_threads", 4, "Number of threads in input pipeline")
flags.DEFINE_boolean("use_cudnn", True, "Whether to use cudnn rnn (should be False for CPU)")
flags.DEFINE_boolean("is_bucket", True, "build bucket batch iterator or not")
flags.DEFINE_boolean("line_limit_prepro", False, "limit prepro to limited number of lines for POC")
flags.DEFINE_boolean("with_passage_ranking", False, "Enable Passage Ranking part")
flags.DEFINE_boolean("visualize_matplot", False, "Save concatenated para length of each query id")
flags.DEFINE_integer("bucket_range", [40, 401, 40], "the range of bucket")

flags.DEFINE_integer("rouge_metric", 0, "# 0 = f, 1 = p, 2 = r")
flags.DEFINE_integer("batch_size", 16, "Batch size") # 64
flags.DEFINE_integer("num_steps", 50000, "Number of steps")
flags.DEFINE_integer("checkpoint", 1000, "checkpoint to save and evaluate the model")
flags.DEFINE_integer("period", 100, "period to save batch loss")
flags.DEFINE_integer("val_num_batches", 150, "Number of batches to evaluate the model")
flags.DEFINE_float("init_lr", 0.5, "Initial learning rate for Adadelta")
flags.DEFINE_float("keep_prob", 0.7, "Dropout keep prob in rnn") #0.7
flags.DEFINE_float("ptr_keep_prob", 0.7, "Dropout keep prob for pointer network") #0.7
flags.DEFINE_float("grad_clip", 5.0, "Global Norm gradient clipping rate")
flags.DEFINE_integer("hidden", 75, "Hidden size") #75
flags.DEFINE_integer("char_hidden", 100, "GRU dimention for char")
flags.DEFINE_integer("patience", 3, "Patience for learning rate decay")

filename = "train.tfrecords"

iterator = tf.python_io.tf_record_iterator(path=train_record_file)
config = flags.FLAGS

def get_record_parser(config, is_test=False):
	def parse(example):
		para_limit = config.test_para_limit if is_test else config.para_limit
		ques_limit = config.test_ques_limit if is_test else config.ques_limit
		char_limit = config.char_limit
		with tf.device('/cpu:0'):
			features = tf.parse_single_example(example,
											   features={
												   "passage_idxs": tf.FixedLenFeature([], tf.string),
												   "ques_idxs": tf.FixedLenFeature([], tf.string),
												   "passage_char_idxs": tf.FixedLenFeature([], tf.string),
												   "ques_char_idxs": tf.FixedLenFeature([], tf.string),
												   "y1": tf.FixedLenFeature([], tf.string),
												   "y2": tf.FixedLenFeature([], tf.string),
												   "id": tf.FixedLenFeature([], tf.int64)
											   })
			passage_idxs = tf.reshape(tf.decode_raw(
				features["passage_idxs"], tf.int32), [para_limit])
			ques_idxs = tf.reshape(tf.decode_raw(
				features["ques_idxs"], tf.int32), [ques_limit])
			passage_char_idxs = tf.reshape(tf.decode_raw(
				features["passage_char_idxs"], tf.int32), [para_limit, char_limit])
			ques_char_idxs = tf.reshape(tf.decode_raw(
				features["ques_char_idxs"], tf.int32), [ques_limit, char_limit])
			y1 = tf.reshape(tf.decode_raw(
				features["y1"], tf.float32), [para_limit])
			y2 = tf.reshape(tf.decode_raw(
				features["y2"], tf.float32), [para_limit])
			qa_id = features["id"]
			return passage_idxs, ques_idxs, passage_char_idxs, ques_char_idxs, y1, y2, qa_id
	return parse
json_dict = {}
sess = tf.Session()
#sess = tf.Session(config=sess_config)
with sess.as_default():
	for i in tqdm(iterator):
		parse = get_record_parser(config)
		a,b,c,d,e,f,g = parse(i)
		with tf.device('/cpu:0'):
			h = tf.count_nonzero(a,dtype=tf.int32)
		qa_id = g.eval()
		a = a.eval()
		h = np.count_nonzero(a)	
		json_dict[int(qa_id)] = int(h)
	with open('para_metadata.json','w') as fp:
		json.dump(concat_para_length,fp)
#matplotlib inline
#x = np.random.normal(size = 1000)
plt.hist(list(json_dict.values()), bins='auto')
plt.title("Histogram with 'auto' bins")
plt.ylabel('para length')
plt.show()