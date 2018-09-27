import tensorflow as tf
from func import cudnn_gru, native_gru, dot_attention, summ, dropout, ptr_net, pr_attention, dense


class Model(object):
	def __init__(self, config, batch, word_mat=None, char_mat=None, trainable=True, opt=True):
		self.config = config
		self.global_step = tf.get_variable('global_step', shape=[], dtype=tf.int32,
										   initializer=tf.constant_initializer(0), trainable=False)
		self.c, self.q, self.ch, self.qh, self.y1, self.y2, self.qa_id, \
			self.c_pr, self.ch_pr, self.pr, self.y1_pr, self.y2_pr = batch.get_next()
		self.is_train = tf.get_variable(
			"is_train", shape=[], dtype=tf.bool, trainable=False)
		self.word_mat = tf.get_variable("word_mat", initializer=tf.constant(
			word_mat, dtype=tf.float32), trainable=False)
		self.char_mat = tf.get_variable(
			"char_mat", char_mat.shape, dtype=tf.float32)

		self.c_mask = tf.cast(self.c, tf.bool)
		self.q_mask = tf.cast(self.q, tf.bool)
		
		# passage ranking line:
		#self.pr_mask = tf.cast(self.p, tf.bool)

		self.c_len = tf.reduce_sum(tf.cast(self.c_mask, tf.int32), axis=1)
		self.q_len = tf.reduce_sum(tf.cast(self.q_mask, tf.int32), axis=1)

		if opt:
			N, CL = config.batch_size, config.char_limit
			self.c_maxlen = tf.reduce_max(self.c_len)

			###
			self.c_maxlen = config.para_limit
			###
			self.q_maxlen = tf.reduce_max(self.q_len)
			self.c = tf.slice(self.c, [0, 0], [N, self.c_maxlen])
			self.q = tf.slice(self.q, [0, 0], [N, self.q_maxlen])
			self.c_mask = tf.slice(self.c_mask, [0, 0], [N, self.c_maxlen])
			self.q_mask = tf.slice(self.q_mask, [0, 0], [N, self.q_maxlen])
			self.ch = tf.slice(self.ch, [0, 0, 0], [N, self.c_maxlen, CL])
			self.qh = tf.slice(self.qh, [0, 0, 0], [N, self.q_maxlen, CL])
			self.y1 = tf.slice(self.y1, [0, 0], [N, self.c_maxlen])
			self.y2 = tf.slice(self.y2, [0, 0], [N, self.c_maxlen])

			# passage ranking
			#print(self.ch_pr.get_shape())
			#print(self.c_pr.get_shape())
			self.c_pr = tf.slice(self.c_pr, [0, 0], [N, config.max_para*config.para_limit])
			self.ch_pr = tf.slice(self.ch_pr, [0, 0, 0], [N, config.max_para*config.para_limit, CL])
		else:
			self.c_maxlen, self.q_maxlen = config.para_limit, config.ques_limit

		self.ch_len = tf.reshape(tf.reduce_sum(
			tf.cast(tf.cast(self.ch, tf.bool), tf.int32), axis=2), [-1])
		self.qh_len = tf.reshape(tf.reduce_sum(
			tf.cast(tf.cast(self.qh, tf.bool), tf.int32), axis=2), [-1])

		self.ready()

		if trainable:
			self.lr = tf.get_variable(
			"lr", shape=[], dtype=tf.float32, trainable=False)
			self.opt = tf.train.AdadeltaOptimizer(
				learning_rate=self.lr, epsilon=1e-6)

			if config.with_passage_ranking:
				##########################################
				grads_ee = self.opt.compute_gradients(self.e_loss)
				gradients_ee, variables_ee = zip(*grads_ee)
				capped_grads_ee, _ = tf.clip_by_global_norm(
					gradients_ee, config.grad_clip)
				self.train_op_ee = self.opt.apply_gradients(
					zip(capped_grads_ee, variables_ee), global_step=self.global_step)
			else:
				grads = self.opt.compute_gradients(self.loss)
				gradients, variables = zip(*grads)
				capped_grads, _ = tf.clip_by_global_norm(
					gradients, config.grad_clip)
				self.train_op = self.opt.apply_gradients(
					zip(capped_grads, variables), global_step=self.global_step)
	def ready(self):
		config = self.config
		N, PL, QL, CL, d, dc, dg = config.batch_size, self.c_maxlen, self.q_maxlen, config.char_limit, config.hidden, config.char_dim, config.char_hidden
		gru = cudnn_gru if config.use_cudnn else native_gru

		gi = []
		att_vP = []
		
		for i in range(config.max_para):
			print(i)
			with tf.variable_scope("emb"+str(i)):
				with tf.variable_scope("char"+str(i)):
					#CL = tf.Print(CL,[CL],message="CL:")
					#PL = tf.Print(PL,[PL],message="PL:")
					#self.ch_pr = tf.Print(self.ch_pr,[self.ch_pr.get_shape()],message="ch_pr:")
					self.ch_pr_ = self.ch_pr[:,i*400:(i+1)*400,:]
					print(self.ch_pr_.get_shape())
					#self.c_pr = tf.reshape(self.c_pr, [N, 12, PL])
					#print(self.ch.get_shape())
					#print(self.ch_pr.get_shape())
					#print(self.c.get_shape())
					#print(self.c_pr.get_shape())
					#self.ch_pr = tf.Print(self.ch_pr,[self.ch_pr[:,2:,:]],message="ch_pr")
					ch_emb = tf.reshape(tf.nn.embedding_lookup(\
						self.char_mat, self.ch_pr_), [N * PL, CL, dc])
					#	self.char_mat, self.ch), [N * PL, CL, dc])
					qh_emb = tf.reshape(tf.nn.embedding_lookup(
						self.char_mat, self.qh), [N * QL, CL, dc])
					ch_emb = dropout(
						ch_emb, keep_prob=config.keep_prob, is_train=self.is_train)
					#ch_emb = tf.Print(ch_emb,[ch_emb],message="ch_emb")
					#qh_emb = tf.Print(qh_emb,[qh_emb],message="qh_emb")
					qh_emb = dropout(
						qh_emb, keep_prob=config.keep_prob, is_train=self.is_train)
					cell_fw = tf.contrib.rnn.GRUCell(dg)
					cell_bw = tf.contrib.rnn.GRUCell(dg)
					_, (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
						cell_fw, cell_bw, ch_emb, self.ch_len, dtype=tf.float32)
					ch_emb = tf.concat([state_fw, state_bw], axis=1)
					_, (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
						cell_fw, cell_bw, qh_emb, self.qh_len, dtype=tf.float32)
					#state_fw = tf.Print(state_fw,[state_fw],message="state_fw")
					#state_bw = tf.Print(state_bw,[state_bw],message="state_bw")
					qh_emb = tf.concat([state_fw, state_bw], axis=1)
					qh_emb = tf.reshape(qh_emb, [N, QL, 2 * dg])
					ch_emb = tf.reshape(ch_emb, [N, PL, 2 * dg])
					#ch_emb = tf.Print(ch_emb,[ch_emb],message="ch_emb")
				with tf.name_scope("word"+str(i)):
					c_emb = tf.nn.embedding_lookup(self.word_mat, self.c_pr[:,i*400:(i+1)*400])
					q_emb = tf.nn.embedding_lookup(self.word_mat, self.q)

				c_emb = tf.concat([c_emb, ch_emb], axis=2)
				q_emb = tf.concat([q_emb, qh_emb], axis=2)

			with tf.variable_scope("encoding"+str(i)):
				rnn = gru(num_layers=3, num_units=d, batch_size=N, input_size=c_emb.get_shape(
				).as_list()[-1], keep_prob=config.keep_prob, is_train=self.is_train)
				c = rnn(c_emb, seq_len=self.c_len)
				q = rnn(q_emb, seq_len=self.q_len)

			with tf.variable_scope("attention"+str(i)):
				qc_att = dot_attention(c, q, mask=self.q_mask, hidden=d,
									   keep_prob=config.keep_prob, is_train=self.is_train)
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
			#att_vP = tf.Print(att_vP,[tf.shape(att_vP)],message="att_vP:")
			"""
			with tf.variable_scope("match"):
				self_att = dot_attention(
					att, att, mask=self.c_mask, hidden=d, keep_prob=config.keep_prob, is_train=self.is_train)
				rnn = gru(num_layers=1, num_units=d, batch_size=N, input_size=self_att.get_shape(
				).as_list()[-1], keep_prob=config.keep_prob, is_train=self.is_train)
				match = rnn(self_att, seq_len=self.c_len)
			"""
		with tf.variable_scope("pointer"):

			# r_Q:
			init = summ(q[:, :, -2 * d:], d, mask=self.q_mask,
						keep_prob=config.ptr_keep_prob, is_train=self.is_train)
			print("rQ:",init.get_shape().as_list())
			pointer = ptr_net(batch=N, hidden=init.get_shape().as_list(
			)[-1], keep_prob=config.ptr_keep_prob, is_train=self.is_train)
			logits1, logits2 = pointer(init, att, d, self.c_mask)

		with tf.variable_scope("predict"):
			outer = tf.matmul(tf.expand_dims(tf.nn.softmax(logits1), axis=2),
							  tf.expand_dims(tf.nn.softmax(logits2), axis=1))
			outer = tf.matrix_band_part(outer, 0, 15)
			self.yp1 = tf.argmax(tf.reduce_max(outer, axis=2), axis=1)
			self.yp2 = tf.argmax(tf.reduce_max(outer, axis=1), axis=1)
			losses = tf.nn.softmax_cross_entropy_with_logits(
				logits=logits1, labels=self.y1)
			losses2 = tf.nn.softmax_cross_entropy_with_logits(
				logits=logits2, labels=self.y2)
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
				with tf.variable_scope("passage-ranking-attention"+str(i)):

					#att_vP = tf.Print(att_vP,[att_vP.get_shape()],message="att_vP:")
					vj_P = att_vP[:,i*400:(i+1)*400,:]
					pr_att = pr_attention(batch=N, hidden=init.get_shape().as_list(
						)[-1], keep_prob=config.keep_prob, is_train=self.is_train)
					r_P = pr_att(init, vj_P, d, self.c_mask)
					#r_P = tf.Print(r_P,[r_P],message="r_p")
					# Wg
					concatenate = tf.concat([init,r_P],axis=1)
					g = tf.nn.tanh(dense(concatenate, hidden=d, use_bias=False, scope="g"+str(i)))
					g_ = dense(g, 1, use_bias=False, scope="g_"+str(i))
					#g = tf.Print(g,[g],message="g")
					if i==0:
						gi = tf.reshape(g_,[N,1])
					else:
						gi = tf.concat([gi,tf.reshape(g_,[N,1])],axis=1)
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
