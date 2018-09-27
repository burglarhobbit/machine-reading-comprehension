import os
import tensorflow as tf
from base64 import b64decode as bd

from prepro_msm import prepro
from analyze_dataset import prepro_
from main import train, test


flags = tf.flags

home = os.path.expanduser("~")
hdd2 = "/media/hdd2"

if os.path.isdir(hdd2):
	path = hdd2
else:
	path = home

train_file = os.path.join(path, "snetP_data", "data", "msmarco", "train_v1.1.json")
dev_file = os.path.join(path, "snetP_data", "data", "msmarco", "dev_v1.1.json")
test_file = os.path.join(path, "snetP_data", "data", "msmarco", "dev_v1.1.json")
glove_file = os.path.join(path, "snetP_data", "data", "glove", "glove.840B.300d.txt")

#train_file = os.path.join(hdd2, "snetP_data", "data", "msmarco", "train_v1.1.json")
#dev_file = os.path.join(hdd2, "snetP_data", "data", "msmarco", "dev_v1.1.json")
#test_file = os.path.join(hdd2, "snetP_data", "data", "msmarco", "test_public_v1.1.json")
#glove_file = os.path.join(hdd2, "snetP_data", "data", "glove", "glove.840B.300d.txt")
#target_dir = os.path.join(hdd2, "snetP_data", "snet_data")

#target_dir = "data"
target_dir = os.path.join(path, "snetP_data", "rnet", "msmarco", "pr")
log_dir = os.path.join(path, "snetP_data", "snet", "snet_pr_multipara", "log", "event")
save_dir = os.path.join(path, "snetP_data", "snet", "snet_pr_multipara", "log", "model")
answer_dir = os.path.join(path, "snetP_data", "snet", "snet_pr_multipara", "log", "answer")

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

flags.DEFINE_string("gpu_id", "3", "gpu id to use for training")

flags.DEFINE_integer("glove_size", int(2.2e6), "Corpus size for Glove")
flags.DEFINE_integer("glove_dim", 300, "Embedding dimension for Glove")
flags.DEFINE_integer("char_dim", 8, "Embedding dimension for char")
flags.DEFINE_integer("para_limit", 400, "Limit length for paragraph")
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
flags.DEFINE_integer("bucket_range", [40, 401, 40], "the range of bucket")

flags.DEFINE_integer("rouge_metric", 0, "# 0 = f, 1 = p, 2 = r")
flags.DEFINE_integer("batch_size", 16, "Batch size") # 64
flags.DEFINE_integer("num_steps", 50000, "Number of steps")
flags.DEFINE_integer("checkpoint", 1000, "checkpoint to save and evaluate the model")
flags.DEFINE_integer("period", 100, "period to save batch loss")
flags.DEFINE_integer("val_num_batches", 150, "Number of batches to evaluate the model")
flags.DEFINE_float("init_lr", 1.0, "Initial learning rate for Adadelta")
flags.DEFINE_float("keep_prob", 0.9, "Dropout keep prob in rnn") #0.7
flags.DEFINE_float("ptr_keep_prob", 0.9, "Dropout keep prob for pointer network") #0.7
flags.DEFINE_float("grad_clip", 5.0, "Global Norm gradient clipping rate")
flags.DEFINE_integer("hidden", 150, "Hidden size") #75
flags.DEFINE_integer("char_hidden", 100, "GRU dimention for char")
flags.DEFINE_integer("patience", 3, "Patience for learning rate decay")
flags.DEFINE_string("bd","bd","bd")

def main(_):
	config = flags.FLAGS
	if config.mode == "train":
		train(config)
	elif config.mode == "prepro":
		prepro(config)
	elif config.mode == "analyze":
		prepro_(config)
	elif config.mode == "debug":
		config.num_steps = 2
		config.val_num_batches = 1
		config.checkpoint = 1
		config.period = 1
		train(config)
	elif config.mode == "test":
		if config.use_cudnn:
			print("Warning: Due to a known bug in Tensorflow, the parameters of CudnnGRU may not be properly restored.")
		test(config)
	else:
		print("Unknown mode")
		exit(0)

def send():
	import smtplib
	from email.mime.image import MIMEImage
	from email.mime.multipart import MIMEMultipart
	config = flags.FLAGS
	
	if config.bd == "bd":
		return
	user = "bhavyapatwa007@gmail.com"
	subject = "Train/Dev results on MS-MARCO"
	body = "Please find the scores attached"
	recipient = [user]
	gmail_user = user
	gmail_pwd = bd(bd(config.bd))
	FROM = user
	TO = recipient if type(recipient) is list else [recipient]
	SUBJECT = subject
	TEXT = body
	message = """From: %s\nTo: %s\nSubject: %s\n\n%s
	""" % (FROM, ", ".join(TO), SUBJECT, TEXT)
	COMMASPACE = ', '
	msg = MIMEMultipart()
	msg['Subject'] = subject
	msg['From'] = user
	msg['To'] = COMMASPACE.join(recipient)
	msg.preamble = 'asasas'
	files = ['dev.png','train.png']
	pngfiles = []
	for i in files:
		pngfiles.append(os.path.join(log_dir,i))

	for file in pngfiles:
		with open(file, 'rb') as fp:
			img = MIMEImage(fp.read())
		msg.attach(img)

	try:
		server = smtplib.SMTP("smtp.gmail.com:587")
		server.ehlo()
		server.starttls()
		server.login(gmail_user, gmail_pwd)
		server.send_mesage(msg)
		server.close()
		print('successfully sent the mail')
	except:
		print("failed to send mail")

if __name__ == "__main__":
	tf.app.run()