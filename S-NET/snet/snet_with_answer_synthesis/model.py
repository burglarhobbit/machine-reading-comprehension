import tensorflow as tf
from func import cudnn_gru, native_gru, dot_attention, summ, dropout, ptr_net, pr_attention, dense
from func import _build_decoder

class Model(object):
	def __init__(self, config, batch, word_mat=None, char_mat=None, trainable=True, opt=True):
		self.config = config
		self.global_step = tf.get_variable('global_step', shape=[], dtype=tf.int32,
										   initializer=tf.constant_initializer(0), trainable=False)
		self.c, self.q, self.fs, self.fe, self.y1, self.y2, self.ans, \
			self.qa_id = batch.get_next()
	
		self.is_train = tf.get_variable(
			"is_train", shape=[], dtype=tf.bool, trainable=False)
		self.word_mat = tf.get_variable("word_mat", initializer=tf.constant(
			word_mat, dtype=tf.float32), trainable=True) # original was false
		self.char_mat = tf.get_variable(
			"char_mat", char_mat.shape, dtype=tf.float32)

		self.c_mask = tf.cast(self.c, tf.bool)
		self.q_mask = tf.cast(self.q, tf.bool)
		self.ans_mask = tf.cast(self.ans, tf.bool)

		# passage ranking line:
		#self.pr_mask = tf.cast(self.p, tf.bool)

		self.c_len = tf.reduce_sum(tf.cast(self.c_mask, tf.int32), axis=1)
		self.q_len = tf.reduce_sum(tf.cast(self.q_mask, tf.int32), axis=1)
		self.ans_len = tf.reduce_sum(tf.cast(self.ans_mask, tf.int32), axis=1)

		if opt:
			N, CL = config.batch_size, config.char_limit
			self.c_maxlen = tf.reduce_max(self.c_len)
			self.q_maxlen = tf.reduce_max(self.q_len)
			self.ans_maxlen = tf.reduce_max(self.ans_len)
			self.c = tf.slice(self.c, [0, 0], [N, self.c_maxlen])
			self.q = tf.slice(self.q, [0, 0], [N, self.q_maxlen])
			self.ans = tf.slice(self.ans, [0, 0], [N, self.ans_maxlen])
			self.c_mask = tf.slice(self.c_mask, [0, 0], [N, self.c_maxlen])
			self.q_mask = tf.slice(self.q_mask, [0, 0], [N, self.q_maxlen])
			self.ans_mask = tf.slice(self.ans_mask, [0, 0], [N, self.ans_maxlen])
			self.y1 = tf.slice(self.y1, [0, 0], [N, self.c_maxlen])
			self.y2 = tf.slice(self.y2, [0, 0], [N, self.c_maxlen])

		else:
			self.c_maxlen, self.q_maxlen = config.para_limit, config.ques_limit

		self.ready()
		self.merged = tf.summary.merge_all()

		if trainable:
			self.lr = tf.get_variable(
			"lr", shape=[], dtype=tf.float32, trainable=False)
			self.opt = tf.train.AdadeltaOptimizer(
				learning_rate=self.lr, epsilon=1e-6)

			grads_ee = self.opt.compute_gradients(self.e_loss)
			gradients_ee, variables_ee = zip(*grads_ee)
			capped_grads_ee, _ = tf.clip_by_global_norm(
				gradients_ee, config.grad_clip)
			self.train_op_ee = self.opt.apply_gradients(
				zip(capped_grads_ee, variables_ee), global_step=self.global_step)

	def ready(self):
		config = self.config
		N, PL, QL, d = config.batch_size, self.c_maxlen, self.q_maxlen, config.hidden
		keep_prob, is_train = config.keep_prob, config.is_train
		gru = cudnn_gru if config.use_cudnn else native_gru

		with tf.variable_scope("emb"):
			with tf.name_scope("word"):
				c = tf.nn.embedding_lookup(self.word_mat, self.c)
				q = tf.nn.embedding_lookup(self.word_mat, self.q)
			c_emb = tf.concat([c, self.fs, self.fe], axis=2)
			q_emb = q 

		with tf.variable_scope("encoding"):
			rnn = gru(num_layers=3, num_units=d, batch_size=N, input_size=c_emb.get_shape(
			).as_list()[-1], keep_prob=config.keep_prob, is_train=self.is_train)
			c_enc, bw_final_state_c = rnn(c_emb, seq_len=self.c_len)
			q_enc, bw_final_state_q = rnn(q_emb, seq_len=self.q_len)
			
			encoder_outputs = tf.concat([c_enc, q_enc],axis=1)
			bw_final_state = (bw_final_state_c,bw_final_state_q)

		with tf.variable_scope("attention"):
			bi_final_hidden = dropout(bw_final_state, keep_prob=keep_prob, is_train=is_train)
			source_sequence_length = tf.add(PL,QL)

			logits, sample_id, final_context_state = _build_decoder(
				encoder_outputs, bi_final_hidden, config, is_train, source_sequence_length,
				target_sequence_length, target_input, embedding_decoder
			)
			"""
			
			qc_att = dot_attention(c, q, mask=self.q_mask, hidden=d,
				keep_prob=config.keep_prob, is_train=self.is_train,
				name_scope="attention_layer")
			rnn = gru(num_layers=1, num_units=d, batch_size=N, input_size=qc_att.get_shape(
			).as_list()[-1], keep_prob=config.keep_prob, is_train=self.is_train)
			att = rnn(qc_att, seq_len=self.c_len)
			# att is the v_P
			if i==0:
				att_vP = att
			else:
				att_vP = tf.concat([att_vP, att], axis=1)
			#att = tf.Print(att,[att],message="att:")
			print("att:",att.get_shape().as_list())
			print("att_vP:",att_vP.get_shape().as_list())
			"""

		with tf.variable_scope("pointer"):

			# r_Q:
			init = summ(q[:, :, -2 * d:], d, mask=self.q_mask,
						keep_prob=config.ptr_keep_prob, is_train=self.is_train)
			print("rQ:",init.get_shape().as_list())
			pointer = ptr_net(batch=N, hidden=init.get_shape().as_list(
			)[-1], keep_prob=config.ptr_keep_prob, is_train=self.is_train)
			logits1, logits2 = pointer(init, att_vP, d, self.c_pr_mask)
			tf.summary.histogram('rQ_init',init)
			tf.summary.histogram('pointer_logits_1',logits1)
			tf.summary.histogram('pointer_logits_2',logits2)

		with tf.variable_scope("predict"):
			outer = tf.matmul(tf.expand_dims(tf.nn.softmax(logits1), axis=2),
							  tf.expand_dims(tf.nn.softmax(logits2), axis=1))
			outer = tf.matrix_band_part(outer, 0, 15)
			self.yp1 = tf.argmax(tf.reduce_max(outer, axis=2), axis=1)
			self.yp2 = tf.argmax(tf.reduce_max(outer, axis=1), axis=1)
			losses = tf.nn.softmax_cross_entropy_with_logits(
				logits=logits1, labels=self.y1_pr)
			losses2 = tf.nn.softmax_cross_entropy_with_logits(
				logits=logits2, labels=self.y2_pr)
			#losses1_2 = tf.reduce_mean(losses1_2, axis=0)
			self.loss = tf.reduce_mean(losses + losses2)

			# print losses
			#condition = tf.greater(self.loss, 11)
			#self.yp1 = tf.where(condition, tf.Print(self.yp1,[self.yp1],message="Yp1:"), self.yp1)
			#self.yp2 = tf.where(condition, tf.Print(self.yp2,[self.yp2],message="Yp2:"), self.yp1)
		
		if config.with_passage_ranking:
			gi = None
			for i in range(config.max_para):
				# Passage ranking
				if i==0:
					with tf.variable_scope("passage-ranking-attention"):

						#att_vP = tf.Print(att_vP,[att_vP.get_shape()],message="att_vP:")
						vj_P = att_vP[:,i*400:(i+1)*400,:]
						pr_att = pr_attention(batch=N, hidden=init.get_shape().as_list()[-1],
							keep_prob=config.keep_prob, is_train=self.is_train,
							name_scope="passage_ranking_att_layer")
						r_P = pr_att(init, vj_P, d, self.c_mask)
						tf.summary.histogram('r_P_'+str(i),r_P)
						#r_P = tf.Print(r_P,[r_P],message="r_p")
						# Wg
						concatenate = tf.concat([init,r_P],axis=1)
						g = tf.nn.tanh(dense(concatenate, hidden=d, use_bias=False, scope="g",
							name_scope="dense_pr_att_layer_1"))
						g_ = dense(g, 1, use_bias=False, scope="g_",
							name_scope="dense_pr_att_layer_2")
						#g = tf.Print(g,[g],message="g")
						if i==0:
							gi = tf.reshape(g_,[N,1])
						else:
							gi = tf.concat([gi,tf.reshape(g_,[N,1])],axis=1)
				else:
					with tf.variable_scope("passage-ranking-attention", reuse=True):
						#att_vP = tf.Print(att_vP,[att_vP.get_shape()],message="att_vP:")
						vj_P = att_vP[:,i*400:(i+1)*400,:]
						pr_att = pr_attention(batch=N, hidden=init.get_shape().as_list()[-1],
							keep_prob=config.keep_prob, is_train=self.is_train,
							name_scope="passage_ranking_att_layer")
						r_P = pr_att(init, vj_P, d, self.c_mask)
						tf.summary.histogram('r_P_'+str(i),r_P)
						#r_P = tf.Print(r_P,[r_P],message="r_p")
						# Wg

						concatenate = tf.concat([init,r_P],axis=1)
						g = tf.nn.tanh(dense(concatenate, hidden=d, use_bias=False, scope="g",
							name_scope="dense_pr_att_layer_1"))
						g_ = dense(g, 1, use_bias=False, scope="g_",
							name_scope="dense_pr_att_layer_2")
						#g = tf.Print(g,[g],message="g")
						if i==0:
							gi = tf.reshape(g_,[N,1])
						else:
							gi = tf.concat([gi,tf.reshape(g_,[N,1])],axis=1)
			tf.summary.histogram('gi',gi)
			#gi_ = tf.convert_to_tensor(gi,dtype=tf.float32)
			#self.gi = tf.nn.softmax(gi_)
			#self.losses3 = tf.nn.softmax_cross_entropy_with_logits(
			#			logits=gi_, labels=tf.reshape(self.pr,[-1,1]))
			self.losses3 = tf.nn.softmax_cross_entropy_with_logits(
						logits=gi, labels=self.pr)
			#self.losses3 = tf.Print(self.losses3,[self.losses3,tf.reduce_max(self.losses3),
			#	tf.reduce_max(self.pr),tf.reduce_max(gi)],message="losses3:")
			self.pr_loss = tf.reduce_mean(self.losses3)
			#self.pr_loss = tf.Print(self.pr_loss,[self.pr_loss])
			self.r = tf.constant(0.8)
			self.e_loss1 = tf.multiply(self.r,self.loss)
			self.e_loss2 = tf.multiply(tf.subtract(tf.constant(1.0),self.r),self.pr_loss)
			self.e_loss = tf.add(self.e_loss1, self.e_loss2)
			
			#self.loss= tf.Print(self.loss,[self.loss],message="ESP:")
			#self.pr_loss = tf.Print(self.pr_loss,[self.pr_loss],message="PR:")
			#self.e_loss = tf.Print(self.e_loss,[self.e_loss],message="EE:")
	def variable_summaries(var):
		"""Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
		with tf.name_scope('summaries'):
			mean = tf.reduce_mean(var)
			tf.summary.scalar('mean', mean)
			with tf.name_scope('stddev'):
				stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
			tf.summary.scalar('stddev', stddev)
			tf.summary.scalar('max', tf.reduce_max(var))
			tf.summary.scalar('min', tf.reduce_min(var))
			tf.summary.histogram('histogram', var)

	def print(self):
		pass

	def get_loss(self):
		return self.loss

	def get_pr_loss(self):
		return self.pr_loss

	def get_e_loss(self):
		return self.e_loss

	def get_global_step(self):
		return self.global_step