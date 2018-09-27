import tensorflow as tf
import ujson as json
import numpy as np
from tqdm import tqdm
import os

from model import Model
from util import get_record_parser, convert_tokens, evaluate, get_batch_dataset, get_dataset
from tensorflow.python import debug as tf_debug


def train(config):

	gpu_options = sess_config = None
	if config.use_cudnn:
		gpu_options = tf.GPUOptions(visible_device_list=config.gpu_id)
		sess_config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
		sess_config.gpu_options.allow_growth = True
	else:
		gpu_options = tf.GPUOptions(visible_device_list="")
		sess_config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)

	with open(config.word_emb_file, "r") as fh:
		word_mat = np.array(json.load(fh), dtype=np.float32)
	with open(config.char_emb_file, "r") as fh:
		char_mat = np.array(json.load(fh), dtype=np.float32)
	with open(config.train_eval_file, "r") as fh:
		train_eval_file = json.load(fh)
	with open(config.dev_eval_file, "r") as fh:
		dev_eval_file = json.load(fh)
	with open(config.dev_meta, "r") as fh:
		meta = json.load(fh)

	dev_total = meta["total"]

	print("Building model...")
	parser = get_record_parser(config)
	train_dataset = get_batch_dataset(config.train_record_file, parser, config)
	dev_dataset = get_dataset(config.dev_record_file, parser, config)
	handle = tf.placeholder(tf.string, shape=[])
	iterator = tf.data.Iterator.from_string_handle(
		handle, train_dataset.output_types, train_dataset.output_shapes)
	train_iterator = train_dataset.make_one_shot_iterator()
	dev_iterator = dev_dataset.make_one_shot_iterator()

	model = Model(config, iterator, word_mat, char_mat)

	loss_save = 100.0
	patience = 0
	lr = config.init_lr

	with tf.Session(config=sess_config) as sess:
		#sess = tf_debug.LocalCLIDebugWrapperSession(sess)
		writer = tf.summary.FileWriter(config.log_dir, graph=tf.get_default_graph())
		writer.add_graph(sess.graph)

		sess.run(tf.global_variables_initializer())
		saver = tf.train.Saver(max_to_keep=config.max_checkpoint_to_keep,
			save_relative_paths=True)
		#saver.restore(sess, tf.train.latest_checkpoint(config.save_dir))
		if config.restore_checkpoint:
			saver.restore(sess, tf.train.latest_checkpoint(config.save_dir_temp))
		train_handle = sess.run(train_iterator.string_handle())
		dev_handle = sess.run(dev_iterator.string_handle())
		sess.run(tf.assign(model.is_train, tf.constant(True, dtype=tf.bool)))
		sess.run(tf.assign(model.lr, tf.constant(lr, dtype=tf.float32)))
		print("Started training")
		for _ in tqdm(range(1, config.num_steps + 1)):
			global_step = sess.run(model.global_step) + 1
			if config.with_passage_ranking:
				if global_step % config.period == 0:					
					summary, loss_esp, loss_pr, loss_ee, train_op_ee = sess.run(
						[model.loss, model.pr_loss, model.e_loss, model.train_op_ee],
						feed_dict={ handle: train_handle})
					loss_sum1 = tf.Summary(value=[tf.Summary.Value(
						tag="model/loss_esp", simple_value=loss_esp), ])
					loss_sum2 = tf.Summary(value=[tf.Summary.Value(
						tag="model/loss_pr", simple_value=loss_pr), ])
					loss_sum3 = tf.Summary(value=[tf.Summary.Value(
						tag="model/loss_ee", simple_value=loss_ee), ])
					writer.add_summary(loss_sum2, global_step)
					writer.add_summary(loss_sum3, global_step)
					writer.add_summary(loss_sum1, global_step)
					writer.add_summary(summary, global_step)
				else:
					loss_esp, loss_pr, loss_ee, train_op_ee = sess.run(
						[model.loss, model.pr_loss, model.e_loss, model.train_op_ee],
						feed_dict={ handle: train_handle})
			else:
				if global_step % config.period == 0:
					summary, loss_esp, train_op = sess.run([model.merged, model.loss, model.train_op],
						feed_dict={ handle: train_handle})
					loss_sum1 = tf.Summary(value=[tf.Summary.Value(
						tag="model/loss_esp", simple_value=loss_esp), ])
					writer.add_summary(loss_sum1, global_step)
					writer.add_summary(summary, global_step)
				else:
					loss_esp, train_op = sess.run([model.loss, model.train_op],
						feed_dict={ handle: train_handle})
			if global_step % config.checkpoint == 0 or global_step in [1,10,50,100,500]:
				sess.run(tf.assign(model.is_train, tf.constant(False, dtype=tf.bool)))
				_, summ = evaluate_batch(
					model, config.val_num_batches, train_eval_file, sess, "train", handle,
					train_handle, config)
				for s in summ:
					writer.add_summary(s, global_step)
				metrics, summ = evaluate_batch(
					model, dev_total // config.batch_size + 1, dev_eval_file, sess, "dev", handle,
					dev_handle, config)
				sess.run(tf.assign(model.is_train, tf.constant(True, dtype=tf.bool)))

				dev_loss = metrics["loss_ee"]
				if dev_loss < loss_save:
					loss_save = dev_loss
					patience = 0
				else:
					patience += 1
				if patience >= config.patience:
					lr /= 2.0
					loss_save = dev_loss
					patience = 0
				sess.run(tf.assign(model.lr, tf.constant(lr, dtype=tf.float32)))
				for s in summ:
					writer.add_summary(s, global_step)
				writer.flush()
				filename = os.path.join(
					config.save_dir, "model_{}.ckpt".format(global_step))
				saver.save(sess, filename)


def evaluate_batch(model, num_batches, eval_file, sess, data_type, handle, str_handle, config):
	answer_dict = {}
	losses_esp = losses_pr = losses_ee = []
	outlier_count = 0
	for _ in tqdm(range(1, num_batches + 1)):
		if config.with_passage_ranking:
			qa_id, loss_esp, loss_pr, loss_ee, yp1, yp2, = sess.run(
				[model.qa_id, model.loss, model.pr_loss, model.e_loss, model.yp1, model.yp2],
				feed_dict={handle: str_handle})
		else:
			qa_id, loss_esp, yp1, yp2, = sess.run(
				[model.qa_id, model.loss, model.yp1, model.yp2],
				feed_dict={handle: str_handle})
		answer_dict_, _, outlier = convert_tokens(
			config, eval_file, qa_id.tolist(), yp1.tolist(), yp2.tolist())
		if outlier:
			outlier_count += 1
			continue
		answer_dict.update(answer_dict_)
		losses_esp.append(loss_esp)
		if config.with_passage_ranking:
			losses_pr.append(loss_pr)
			losses_ee.append(loss_ee)
	#print("outlier_count:",outlier_count)
	loss_esp = np.mean(losses_esp)
	if config.with_passage_ranking:
		loss_pr = np.mean(losses_pr)
		loss_ee = np.mean(losses_ee)
	metrics = evaluate(eval_file, answer_dict)
	metrics["loss_esp"] = loss_esp
	metrics["loss_ee"] = loss_esp
	if config.with_passage_ranking:
		metrics["loss_pr"] = loss_pr
		metrics["loss_ee"] = loss_ee
	loss_sum1 = tf.Summary(value=[tf.Summary.Value(
		tag="{}/loss_esp".format(data_type), simple_value=metrics["loss_esp"]), ])
	if config.with_passage_ranking:
		loss_sum2 = tf.Summary(value=[tf.Summary.Value(
			tag="{}/loss_pr".format(data_type), simple_value=metrics["loss_pr"]), ])
		loss_sum3 = tf.Summary(value=[tf.Summary.Value(
			tag="{}/loss_ee".format(data_type), simple_value=metrics["loss_ee"]), ])
	f1_sum = tf.Summary(value=[tf.Summary.Value(
		tag="{}/f1".format(data_type), simple_value=metrics["f1"]), ])
	em_sum = tf.Summary(value=[tf.Summary.Value(
		tag="{}/em".format(data_type), simple_value=metrics["exact_match"]), ])
	rouge_l_f = tf.Summary(value=[tf.Summary.Value(
		tag="{}/ROUGE-L".format(data_type), simple_value=metrics["rouge-l-f"]), ])
	rouge_l_p = tf.Summary(value=[tf.Summary.Value(
		tag="{}/rouge-l-p".format(data_type), simple_value=metrics["rouge-l-p"]), ])
	rouge_l_r = tf.Summary(value=[tf.Summary.Value(
		tag="{}/rouge-l-r".format(data_type), simple_value=metrics["rouge-l-r"]), ])
	outlier_c = tf.Summary(value=[tf.Summary.Value(
		tag="{}/outlier_count".format(data_type), simple_value=outlier_count), ])
	if config.with_passage_ranking:
		return metrics, [loss_sum1, loss_sum2, loss_sum3, rouge_l_f]
	return metrics, [loss_sum1, rouge_l_f]

def test(config):

	gpu_options = tf.GPUOptions(visible_device_list="2")
	sess_config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
	sess_config.gpu_options.allow_growth = True

	with open(config.word_emb_file, "r") as fh:
		word_mat = np.array(json.load(fh), dtype=np.float32)
	with open(config.char_emb_file, "r") as fh:
		char_mat = np.array(json.load(fh), dtype=np.float32)
	with open(config.test_eval_file, "r") as fh:
		eval_file = json.load(fh)
	with open(config.test_meta, "r") as fh:
		meta = json.load(fh)

	total = meta["total"]

	print("Loading model...")
	test_batch = get_dataset(config.test_record_file, get_record_parser(
		config, is_test=True), config).make_one_shot_iterator()

	model = Model(config, test_batch, word_mat, char_mat, trainable=False)

	with tf.Session(config=sess_config) as sess:
		sess.run(tf.global_variables_initializer())
		saver = tf.train.Saver()
		saver.restore(sess, tf.train.latest_checkpoint(config.save_dir))
		sess.run(tf.assign(model.is_train, tf.constant(False, dtype=tf.bool)))
		losses = []
		answer_dict = {}
		remapped_dict = {}
		for step in tqdm(range(total // config.batch_size + 1)):
			qa_id, loss, yp1, yp2 = sess.run(
				[model.qa_id, model.loss, model.yp1, model.yp2])
			answer_dict_, remapped_dict_, outlier = convert_tokens(
				eval_file, qa_id.tolist(), yp1.tolist(), yp2.tolist())
			answer_dict.update(answer_dict_)
			remapped_dict.update(remapped_dict_)
			losses.append(loss)
		loss = np.mean(losses)

		# evaluate with answer_dict, but in evaluate-v1.1.py, evaluate with remapped_dict
		# since only that is saved. Both dict are a little bit different, check evaluate-v1.1.py
		metrics = evaluate(eval_file, answer_dict)
		with open(config.answer_file, "w") as fh:
			json.dump(remapped_dict, fh)
		print("Exact Match: {}, F1: {} Rouge-L-F1: {} Rouge-L-p: {} Rouge-l-r: {}".format(\
			metrics['exact_match'], metrics['f1'], metrics['rouge-l-f'], metrics['rouge-l-p'],\
			metrics['rouge-l-r']))