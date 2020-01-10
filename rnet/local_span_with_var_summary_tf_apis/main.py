import tensorflow as tf
import ujson as json
import numpy as np
from tqdm import tqdm
import os

from model import Model
from util import get_record_parser, convert_tokens, evaluate, get_batch_dataset, get_dataset
from tensorflow.python import debug as tf_debug
from analyze_dataset import word_tokenize
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

def train(config):

	gpu_options = tf.GPUOptions(visible_device_list=config.gpu_id)
	sess_config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
	sess_config.gpu_options.allow_growth = True

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
		sess = tf_debug.LocalCLIDebugWrapperSession(sess)
		writer = tf.summary.FileWriter(config.log_dir, graph=tf.get_default_graph())
		writer.add_graph(sess.graph)
		
		sess.run(tf.global_variables_initializer())
		saver = tf.train.Saver(max_to_keep=config.max_checkpoint_to_keep,
			save_relative_paths=True)
		#print(config.save_dir_temp)
		if config.restore_checkpoint:
			saver.restore(sess, tf.train.latest_checkpoint(config.save_dir_temp))
		#saver.restore(sess, tf.train.latest_checkpoint(config.save_dir))
		train_handle = sess.run(train_iterator.string_handle())
		dev_handle = sess.run(dev_iterator.string_handle())
		sess.run(tf.assign(model.is_train, tf.constant(True, dtype=tf.bool)))
		sess.run(tf.assign(model.lr, tf.constant(lr, dtype=tf.float32)))
		print("Started training")
		for _ in tqdm(range(1, config.num_steps + 1)):
			global_step = sess.run(model.global_step) + 1
			summary, loss, train_op = sess.run([model.merged, model.loss, model.train_op], feed_dict={
									  handle: train_handle})
			if global_step % config.period == 0:
				loss_sum = tf.Summary(value=[tf.Summary.Value(
					tag="model/loss", simple_value=loss), ])
				writer.add_summary(loss_sum, global_step)
				writer.add_summary(summary, global_step)
			#print(global_step)
			if global_step % config.checkpoint == 0 or global_step in [1,10,50,100,500]:
				sess.run(tf.assign(model.is_train,
								   tf.constant(False, dtype=tf.bool)))
				_, summ = evaluate_batch(
					model, config.val_num_batches, train_eval_file, sess, "train", handle, train_handle)
				for s in summ:
					writer.add_summary(s, global_step)
				metrics, summ = evaluate_batch(
					model, dev_total // config.batch_size + 1, dev_eval_file, sess, "dev", handle, dev_handle)
				sess.run(tf.assign(model.is_train,
								   tf.constant(True, dtype=tf.bool)))

				dev_loss = metrics["loss"]
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


def evaluate_batch(model, num_batches, eval_file, sess, data_type, handle, str_handle):
	answer_dict = {}
	losses = []
	outlier_count = 0
	for _ in tqdm(range(1, num_batches + 1)):
		qa_id, loss, yp1, yp2, = sess.run(
			[model.qa_id, model.loss, model.yp1, model.yp2], feed_dict={handle: str_handle})
		answer_dict_, _, outlier = convert_tokens(
			eval_file, qa_id.tolist(), yp1.tolist(), yp2.tolist())
		#if outlier:
		#	outlier_count += 1
		#	continue
		answer_dict.update(answer_dict_)
		losses.append(loss)
	#print("outlier_count:",outlier_count)
	loss = np.mean(losses)
	metrics = evaluate(eval_file, answer_dict)
	metrics["loss"] = loss
	loss_sum = tf.Summary(value=[tf.Summary.Value(
		tag="{}/loss".format(data_type), simple_value=metrics["loss"]), ])
	f1_sum = tf.Summary(value=[tf.Summary.Value(
		tag="{}/f1".format(data_type), simple_value=metrics["f1"]), ])
	em_sum = tf.Summary(value=[tf.Summary.Value(
		tag="{}/em".format(data_type), simple_value=metrics["exact_match"]), ])
	rouge_l_f = tf.Summary(value=[tf.Summary.Value(
		tag="{}/ROUGE-L-F1".format(data_type), simple_value=metrics["rouge-l-f"]), ])

	return metrics, [loss_sum, f1_sum, em_sum, rouge_l_f]


def test(config):
	from numpy import concatenate as concat
	gpu_options = tf.GPUOptions(visible_device_list=config.gpu_id)
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
		if config.is_debug:
			sess = tf_debug.LocalCLIDebugWrapperSession(sess)
		sess.run(tf.global_variables_initializer())
		saver = tf.train.Saver()
		saver.restore(sess, tf.train.latest_checkpoint(config.save_dir))
		sess.run(tf.assign(model.is_train, tf.constant(False, dtype=tf.bool)))
		losses = []
		answer_dict = {}
		remapped_dict = {}

		#
		#fh = open('dev_qid_question_map.json','r').read()
		#qa_id_map = json.loads(fh)
		#
		# tqdm
		##########
		# saving {attention,match}_{logits,outputs}, {start,end}_logits,
		start_concat = end_concat = match_logits_concat = match_outputs_concat = \
		att_logits_concat = att_outputs_concat = None
		qa_id_concat = []
		##########
		#for step in tqdm(range(total // config.batch_size + 1)):
		for step in tqdm(range(3)):
			start, end, match_logits, match_outputs, att_logits, att_outputs, \
				qa_id, loss, yp1, yp2 = sess.run(
				[model.predict_outer_start, model.predict_outer_end,
				model.match_logits, model.match_outputs,
				model.att_logits, model.att_outputs,
				model.qa_id, model.loss, model.yp1, model.yp2])
			qa_id = qa_id.tolist()
			yp1 = yp1.tolist()
			yp2 = yp2.tolist()
			answer_dict_, remapped_dict_, outlier = convert_tokens(
				eval_file, qa_id, yp1, yp2)
			answer_dict.update(answer_dict_)
			remapped_dict.update(remapped_dict_)
			losses.append(loss)
			print("\nloss:",loss)
			i = 0
			for qa_id_,yp1_,yp2_ in zip(qa_id,yp1,yp2):
				eval_file_temp = eval_file[str(qa_id_)]
				start_ = eval_file_temp["spans"][yp1_][0]
				end_ = eval_file_temp["spans"][yp2_][1]
				passage = eval_file_temp["passage_concat"][start_:end_]
				qid = eval_file_temp["uuid"]
				qid = str(qid)
				qa_id_concat.append(int(qid))
				passage_token = word_tokenize(passage)
				#print(list(eval_file_temp.keys()))
				question = eval_file_temp["question"]
				question_token = word_tokenize(question)
				example = att_logits[i][yp1_:yp2_+1,:]
				argmax = np.argmax(example,axis=1)
				#print(example)
				#print(example.shape)
				#print(passage_token)
				#print(question_token)
				
				i += 1
				#abcd = input("pause...")
				#demo()
			print(start.shape)
			print(end.shape)
			print(match_logits.shape)
			print(match_outputs.shape)
			print(att_logits.shape)
			print(att_outputs.shape)
			st = start.shape
			et = end.shape
			mlt = match_logits.shape
			mot = match_outputs.shape
			alt = att_logits.shape
			aot = att_outputs.shape
			bs = config.batch_size
			pl = config.para_limit
			ql = 25 # question limit
			start_temp = np.zeros((bs,pl),dtype=start.dtype)
			end_temp = np.zeros((bs,pl),dtype=end.dtype)
			match_logits_temp = np.zeros((bs,pl,pl),dtype=match_logits.dtype)
			match_outputs_temp = np.zeros((bs,pl,mot[2]),dtype=match_logits.dtype)
			att_logits_temp = np.zeros((bs,pl,ql),dtype=match_logits.dtype)
			att_outputs_temp = np.zeros((bs,pl,aot[2]),dtype=match_logits.dtype)
			start_temp[:,:st[1]] = start
			end_temp[:,:et[1]] = end
			match_logits_temp[:,:mlt[1],:mlt[2]] = match_logits
			match_outputs_temp[:,:mot[1],:mot[2]] = match_outputs
			att_logits_temp[:,:alt[1],:alt[2]] = att_logits
			att_outputs_temp[:,:aot[1],:aot[2]] = att_outputs
			if step == 0:
				start_concat = start_temp
				end_concat = end_temp
				match_logits_concat = match_logits_temp
				match_outputs_concat = match_outputs_temp
				att_logits_concat = att_logits_temp
				att_outputs_concat = att_outputs_temp
			else:
				# default axis 0 
				start_concat = concat((start_concat,start_temp),axis=0)
				end_concat = concat((end_concat,end_temp),axis=0)
				match_logits_concat = concat((match_logits_concat,match_logits_temp),axis=0)
				match_outputs_concat = concat((match_outputs_concat,match_outputs_temp),axis=0)
				att_logits_concat = concat((att_logits_concat,att_logits_temp),axis=0)
				att_outputs_concat = concat((att_outputs_concat,att_outputs_temp),axis=0)
			#if(loss>50):
			#	for i,j,k in zip(qa_id.tolist(),yp1.tolist(),yp2.tolist()):
			#		print(answer_dict[str(i)],j,k)
			#	#print("IDs: {} Losses: {} Yp1: {} Yp2: {}".format(qa_id.tolist(),\
			#	#	loss.tolist(), yp1.tolist(), yp2.tolist()))
		print("Saving variables...")
		qa_id_concat = np.array(qa_id_concat)
		save_path = os.path.join(config.answer_dir,'variables.npz')
		np.savez_compressed(save_path,start=start_concat,end=end_concat,
			match_logits=match_logits_concat,match_outputs=match_outputs_concat,
			att_logits=att_logits_concat,att_outputs=att_outputs_concat,
			qa_id=qa_id_concat)

		loss = np.mean(losses)

		# evaluate with answer_dict, but in evaluate-v1.1.py, evaluate with remapped_dict
		# since only that is saved. Both dict are a little bit different, check evaluate-v1.1.py
		print("Saving answer file...")
		metrics = evaluate(eval_file, answer_dict)
		with open(config.answer_file, "w") as fh:
			json.dump(remapped_dict, fh)
		print("Exact Match: {}, F1: {} Rouge-l-f: {} Rouge-l-p: {} Rouge-l-r: {}".format(\
			metrics['exact_match'], metrics['f1'], metrics['rouge-l-f'], metrics['rouge-l-p'],\
			metrics['rouge-l-r']))

def get_demo_image():
    import numpy as np
    from matplotlib.cbook import get_sample_data
    f = get_sample_data("axes_grid/bivariate_normal.npy", asfileobj=False)
    z = np.load(f)
    # z is a numpy array of 15x15
    return z, (-3, 4, -4, 3)


def demo_simple_image(ax):
    Z, extent = get_demo_image()

    im = ax.imshow(Z, extent=extent, interpolation="nearest")
    cb = plt.colorbar(im)
    plt.setp(cb.ax.get_yticklabels(), visible=False)


def demo_locatable_axes_hard(fig1):

    from mpl_toolkits.axes_grid1 \
        import SubplotDivider, LocatableAxes, Size

    divider = SubplotDivider(fig1, 2, 2, 2, aspect=True)

    # axes for image
    ax = LocatableAxes(fig1, divider.get_position())

    # axes for colorbar
    ax_cb = LocatableAxes(fig1, divider.get_position())

    h = [Size.AxesX(ax),  # main axes
         Size.Fixed(0.05),  # padding, 0.1 inch
         Size.Fixed(0.2),  # colorbar, 0.3 inch
         ]

    v = [Size.AxesY(ax)]

    divider.set_horizontal(h)
    divider.set_vertical(v)

    ax.set_axes_locator(divider.new_locator(nx=0, ny=0))
    ax_cb.set_axes_locator(divider.new_locator(nx=2, ny=0))

    fig1.add_axes(ax)
    fig1.add_axes(ax_cb)

    ax_cb.axis["left"].toggle(all=False)
    ax_cb.axis["right"].toggle(ticks=True)

    Z, extent = get_demo_image()

    im = ax.imshow(Z, extent=extent, interpolation="nearest")
    plt.colorbar(im, cax=ax_cb)
    plt.setp(ax_cb.get_yticklabels(), visible=False)


def demo_locatable_axes_easy(ax):
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    divider = make_axes_locatable(ax)

    ax_cb = divider.new_horizontal(size="5%", pad=0.05)
    fig1 = ax.get_figure()
    fig1.add_axes(ax_cb)

    Z, extent = get_demo_image()
    im = ax.imshow(Z, extent=extent, interpolation="nearest")

    plt.colorbar(im, cax=ax_cb)
    ax_cb.yaxis.tick_right()
    for tl in ax_cb.get_yticklabels():
        tl.set_visible(False)
    ax_cb.yaxis.tick_right()


def demo_images_side_by_side(ax):
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    divider = make_axes_locatable(ax)

    Z, extent = get_demo_image()
    ax2 = divider.new_horizontal(size="100%", pad=0.05)
    fig1 = ax.get_figure()
    fig1.add_axes(ax2)

    ax.imshow(Z, extent=extent, interpolation="nearest")
    ax2.imshow(Z, extent=extent, interpolation="nearest")
    for tl in ax2.get_yticklabels():
        tl.set_visible(False)


def demo():

    fig1 = plt.figure(1, (6, 6))
    fig1.clf()

    # PLOT 1
    # simple image & colorbar
    ax = fig1.add_subplot(2, 2, 1)
    demo_simple_image(ax)

    # PLOT 2
    # image and colorbar whose location is adjusted in the drawing time.
    # a hard way

    demo_locatable_axes_hard(fig1)

    # PLOT 3
    # image and colorbar whose location is adjusted in the drawing time.
    # a easy way

    ax = fig1.add_subplot(2, 2, 3)
    demo_locatable_axes_easy(ax)

    # PLOT 4
    # two images side by side with fixed padding.

    ax = fig1.add_subplot(2, 2, 4)
    demo_images_side_by_side(ax)

    plt.draw()
    plt.show()