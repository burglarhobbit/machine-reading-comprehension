import tensorflow as tf
from func import cudnn_gru, native_gru, dot_attention, summ, dropout, ptr_net, pr_attention, dense


class Model(object):
	def __init__(self, config, batch, word_mat=None, char_mat=None, trainable=True, opt=True):
		self.config = config
		self.global_step = tf.get_variable('global_step', shape=[], dtype=tf.int32,
										   initializer=tf.constant_initializer(0), trainable=False)
		self.c, self.q, self.ch, self.qh, self.y1, self.y2, self.qa_id, \
			self.c_pr, self.ch_pr, self.pr, self.para_count, self.answer_info = batch.get_next()
		self.para_count = tf.cast(self.para_count,tf.int32)
		self.i = tf.constant(0, dtype=tf.int32)
		
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
			#self.c_maxlen = config.para_limit
			###
			self.q_maxlen = tf.reduce_max(self.q_len)
			self.c = tf.slice(self.c, [0, 0], [N, self.c_maxlen*config.max_para])
			self.q = tf.slice(self.q, [0, 0], [N, self.q_maxlen])
			self.c_mask = tf.slice(self.c_mask, [0, 0], [N, self.c_maxlen])
			self.q_mask = tf.slice(self.q_mask, [0, 0], [N, self.q_maxlen])
			self.ch = tf.slice(self.ch, [0, 0, 0], [N, self.c_maxlen, CL])
			self.qh = tf.slice(self.qh, [0, 0, 0], [N, self.q_maxlen, CL])
			#self.y1 = tf.slice(self.y1, [0, 0], [N, self.c_maxlen])
			#self.y2 = tf.slice(self.y2, [0, 0], [N, self.c_maxlen])

			# passage ranking
			#print(self.ch_pr.get_shape())
			#print(self.c_pr.get_shape())
			self.c_pr_mask = tf.cast(self.c_pr, tf.bool)
			#self.c_pr_mask = tf.slice(self.c_pr_mask, [0, 0], [N, config.max_para*config.para_limit])
			###
			###
			self.ch_pr = tf.slice(self.ch_pr, [0, 0, 0], [N, config.max_para*config.para_limit, CL])
		else:
			self.c_maxlen, self.q_maxlen = config.para_limit, config.ques_limit

		self.ch_len = tf.reshape(tf.reduce_sum(
			tf.cast(tf.cast(self.ch, tf.bool), tf.int32), axis=2), [-1])
		self.qh_len = tf.reshape(tf.reduce_sum(
			tf.cast(tf.cast(self.qh, tf.bool), tf.int32), axis=2), [-1])

		self.rnn1 = None
		self.rnn2 = None
		self.att_vP = None
		self.ready()
		self.merged = tf.summary.merge_all()

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

	def get_vP(self,i,att_vP,q_,answer_info,y1,y2,c_pr_mask,cmax_c,clen_c):
		# max para limit
		config = self.config

		opt = True
		MPL = config.para_limit
		zero = tf.constant(0, dtype=tf.int32)
		j = tf.constant(0, dtype=tf.int32)

		c = self.c_pr[:,i*MPL:(i+1)*MPL]
		ch = self.ch_pr[:,i*MPL:(i+1)*MPL,:]
		qh = self.qh
		q = self.q

		c_mask = tf.cast(c, tf.bool)
		q_mask = self.q_mask
		
		# passage ranking line:
		#self.pr_mask = tf.cast(self.p, tf.bool)

		c_len = tf.reduce_sum(tf.cast(c_mask, tf.int32), axis=1)
		c_len_int = tf.reshape(c_len,[config.batch_size,1])
		q_len = self.q_len

		if opt:
			N, CL = config.batch_size, config.char_limit
			c_maxlen = tf.reduce_max(c_len)
			c_maxlen_int = tf.reshape(tf.reduce_max(c_len_int),[1])
			q_maxlen = q_len
			c = tf.slice(c, [0, 0], [N, c_maxlen])
			c_mask = tf.slice(c_mask, [0, 0], [N, c_maxlen])
			q_mask = self.q_mask
			ch = tf.slice(ch, [0, 0, 0], [N, c_maxlen, CL])
			qh = self.qh

			temp = self.y2[:,i*MPL:(i+1)*MPL]
			#self.y1 = tf.Print(self.y1,["y1:",tf.shape(self.y1)])
			#self.y2 = tf.Print(self.y2,["y2:",tf.shape(self.y2)])
			y1__ = tf.slice(self.y1, [0, i*MPL], [N, c_maxlen])
			#y1__ = tf.Print(y1__,["y1__:",tf.shape(y1__)])
			
			y2__ = tf.slice(self.y2, [0, i*MPL], [N, c_maxlen])

			def b1(): return c_mask
			def b2(): return tf.concat([c_pr_mask, c_mask],axis=1)
			c_pr_mask = tf.cond(tf.equal(i, zero), b1, b2)

			def b3(): return c_maxlen_int, c_len_int
			def b4():
				print(clen_c.get_shape(),c_len_int.get_shape())
				a = tf.concat([cmax_c, c_maxlen_int],axis=0)
				b = tf.concat([clen_c, c_len_int],axis=1)
				return a,b
			cmax_c, clen_c = tf.cond(tf.equal(i, zero), b3, b4)
			# passage ranking
			#print(self.ch_pr.get_shape())
			#print(self.c_pr.get_shape())
			#c_pr_mask = tf.cast(self.c_pr, tf.bool)
			#c_pr_mask = tf.slice(self.c_pr_mask, [0, i*MPL], [N, c_maxlen])
			###
			###
			#ch_pr = tf.slice(self.ch_pr, [0, i*MPL, 0], [N, c_maxlen, CL])
		else:
			self.c_maxlen, self.q_maxlen = config.para_limit, config.ques_limit

		ch_len = tf.reshape(tf.reduce_sum(
			tf.cast(tf.cast(ch, tf.bool), tf.int32), axis=2), [-1])
		qh_len = self.qh_len
		
		config = self.config
		N, PL, QL, CL, d, dc, dg = config.batch_size, c_maxlen, self.q_maxlen, \
			config.char_limit, config.hidden, config.char_dim, config.char_hidden
		gru = cudnn_gru if config.use_cudnn_gru else native_gru

		with tf.variable_scope("emb"):
			with tf.variable_scope("char"):
				#CL = tf.Print(CL,[CL],message="CL:")
				#PL = tf.Print(PL,[PL],message="PL:")
				#self.ch_pr = tf.Print(self.ch_pr,[self.ch_pr.get_shape()],message="ch_pr:")
				#self.c_pr = tf.reshape(self.c_pr, [N, 12, PL])
				#print(self.ch.get_shape())
				#print(self.ch_pr.get_shape())
				#print(self.c.get_shape())
				#print(self.c_pr.get_shape())
				#self.ch_pr = tf.Print(self.ch_pr,[self.ch_pr[:,2:,:]],message="ch_pr")
				ch_emb = tf.reshape(tf.nn.embedding_lookup(\
					self.char_mat, ch), [N * PL, CL, dc])
				#	self.char_mat, self.ch), [N * PL, CL, dc])
				print(ch.shape,PL)
				print(qh.shape,QL)
				qh_emb = tf.reshape(tf.nn.embedding_lookup(\
					self.char_mat, qh), [N * QL, CL, dc])
				ch_emb = dropout(
					ch_emb, keep_prob=config.keep_prob, is_train=self.is_train)
				#ch_emb = tf.Print(ch_emb,[ch_emb],message="ch_emb")
				#qh_emb = tf.Print(qh_emb,[qh_emb],message="qh_emb")
				qh_emb = dropout(
					qh_emb, keep_prob=config.keep_prob, is_train=self.is_train)
				_, (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
					self.cell_fw, self.cell_bw, ch_emb, ch_len, dtype=tf.float32)
				ch_emb = tf.concat([state_fw, state_bw], axis=1)
				_, (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
					self.cell_fw, self.cell_bw, qh_emb, qh_len, dtype=tf.float32)
				#state_fw = tf.Print(state_fw,[state_fw],message="state_fw")
				#state_bw = tf.Print(state_bw,[state_bw],message="state_bw")
				qh_emb = tf.concat([state_fw, state_bw], axis=1)
				qh_emb = tf.reshape(qh_emb, [N, QL, 2 * dg])
				ch_emb = tf.reshape(ch_emb, [N, PL, 2 * dg])
				#ch_emb = tf.Print(ch_emb,[ch_emb],message="ch_emb")
			with tf.name_scope("word"):
				c_emb = tf.nn.embedding_lookup(self.word_mat, c)
				q_emb = tf.nn.embedding_lookup(self.word_mat, q)

			c_emb = tf.concat([c_emb, ch_emb], axis=2)
			q_emb = tf.concat([q_emb, qh_emb], axis=2)
	
		with tf.variable_scope("encoding", reuse=tf.AUTO_REUSE):
			"""
			def f1():
				self.rnn1 = gru(num_layers=3, num_units=d, batch_size=N, input_size=c_emb.get_shape(
				).as_list()[-1], keep_prob=config.keep_prob, is_train=self.is_train)
				return self.rnn1(c_emb, seq_len=self.c_len)
			def f2():
				return self.rnn1(c_emb, seq_len=self.c_len)
			c = tf.cond(tf.equal(i, zero), f1, f2)
			#q = tf.cond(tf.equal(i, zero), f1, f2)
			#c = rnn(c_emb, seq_len=self.c_len)
			q = self.rnn1(q_emb, seq_len=self.q_len)
			self.q_enc = q
			#self.rnn1 = rnn
			"""
			rnn = gru(num_layers=3, num_units=d, batch_size=N, input_size=c_emb.get_shape(
				).as_list()[-1], keep_prob=config.keep_prob, is_train=self.is_train)

			c = rnn(c_emb, seq_len=c_len)
			q = rnn(q_emb, seq_len=q_len)
			#c_len = tf.Print(c_len,[c_len,tf.shape(c)],message="C:")
			#self.q_enc = q
			q__ = q

		with tf.variable_scope("attention", reuse=tf.AUTO_REUSE):
			qc_att = dot_attention(c, q, mask=q_mask, hidden=d,
				keep_prob=config.keep_prob, is_train=self.is_train,
				name_scope="attention_layer")
			"""
			print("qc_att:",qc_att.shape)
			def f3():
				self.rnn2 = gru(num_layers=1, num_units=d, batch_size=N, input_size=qc_att.get_shape(
				).as_list()[-1], keep_prob=config.keep_prob, is_train=self.is_train)
				return self.rnn2(qc_att, seq_len=self.c_len)
			def f4():
				return self.rnn2(qc_att, seq_len=self.c_len)
			att = tf.cond(tf.equal(self.i, zero), f3, f4)
			"""
			rnn = gru(num_layers=1, num_units=d, batch_size=N, input_size=qc_att.get_shape(
				).as_list()[-1], keep_prob=config.keep_prob, is_train=self.is_train)
			att = rnn(qc_att, seq_len=c_len)
			###
			#att = tf.Print(att,[tf.greater(tf.cast(tf.shape(att)[1],tf.int64),y1_),
			#	tf.shape(att)],message="att:")
			def f5():	
				return att
			def f6():
				return tf.concat([att_vP, att], axis=1)
			#att = rnn(qc_att, seq_len=self.c_len)
			#self.rnn2 = rnn
			# att is the v_P
			att_vP = tf.cond(tf.equal(i, zero), f5, f6)

		def f7(): return y1__,y2__
		def f8(): return tf.concat([y1, y1__], axis=1),tf.concat([y2, y2__], axis=1)
		y1,y2 = tf.cond(tf.equal(i, zero), f7, f8)	
		
		return tf.add(i,tf.constant(1)), att_vP, q__,answer_info,y1,y2,c_pr_mask,cmax_c,clen_c
	
	def condition(self,i,att_vP,q,answer_info,y1,y2,c_pr_mask,cm,cl):
		max_para = tf.reduce_max(self.para_count)
		return tf.less(i, max_para)
	
	def ready(self):
		config = self.config
		N, PL, QL, CL, d, dc, dg = config.batch_size, self.c_maxlen, self.q_maxlen, \
			config.char_limit, config.hidden, config.char_dim, config.char_hidden
		gru = cudnn_gru if config.use_cudnn_gru else native_gru
		
		gi = []
		#att_vP = []
		
		self.cell_fw = tf.contrib.rnn.GRUCell(dg)
		self.cell_bw = tf.contrib.rnn.GRUCell(dg)
		self.rnn1 = None
		self.rnn2 = None
		self.att_vP = tf.zeros([N, 1, 2*d])
		c_pr_mask = self.c_pr_mask
		qtemp = tf.zeros([N, 1, 900])
		
		#   _c = concatenation
		cmax_c = tf.zeros([5], tf.int32)
		clen_c = tf.zeros([N,5], tf.int32)
		"""
		self.rnn1 = gru(num_layers=3, num_units=d, batch_size=N, input_size=500,\
			keep_prob=config.keep_prob, is_train=self.is_train)
		self.rnn2 = gru(num_layers=1, num_units=d, batch_size=N, input_size=1800,\
			keep_prob=config.keep_prob, is_train=self.is_train)
		"""
		result, self.att_vP, q, self.answer_info, self.y1, self.y2, self.c_pr_mask, cmax_c, clen_c = \
			tf.while_loop(self.condition, self.get_vP, loop_vars=[self.i,self.att_vP,qtemp, \
			self.answer_info,self.y1,self.y2,c_pr_mask, cmax_c, clen_c], shape_invariants= \
			[self.i.get_shape(), tf.TensorShape([N, None, 2*d]), tf.TensorShape([N, None, 900]), \
			self.answer_info.get_shape(), tf.TensorShape([None, None]), tf.TensorShape([None, None]), \
			tf.TensorShape([None, None]), tf.TensorShape([None]), tf.TensorShape([N, None])])

		tf.summary.histogram('att_vP',self.att_vP)
		#att_vP = tf.Print(att_vP,[tf.shape(att_vP)],message="att_vP:")
		"""
		with tf.variable_scope("match"):
			self_att = dot_attention(
				att, att, mask=self.c_mask, hidden=d,
				keep_prob=config.keep_prob, is_train=self.is_train)
			rnn = gru(num_layers=1, num_units=d, batch_size=N, input_size=self_att.get_shape(
			).as_list()[-1], keep_prob=config.keep_prob, is_train=self.is_train)
			match = rnn(self_att, seq_len=self.c_len)
		"""
		with tf.variable_scope("pointer"):

			# r_Q:
			#self.att_vP = tf.Print(self.att_vP,[tf.shape(self.att_vP),tf.shape(self.c_pr_mask)],
			#	message="pointer:")
			#self.att_vP = tf.Print(self.att_vP,[tf.greater(self.att_vP,y1),tf.shape(self.c_mask)],
			#	message="pointer:")
			init = summ(q[:, :, -2 * d:], d, mask=self.q_mask,
						keep_prob=config.ptr_keep_prob, is_train=self.is_train)
			print("rQ:",init.get_shape().as_list())
			pointer = ptr_net(batch=N, hidden=init.get_shape().as_list(
			)[-1], keep_prob=config.ptr_keep_prob, is_train=self.is_train)
			logits1, logits2 = pointer(init, self.att_vP, d, self.c_pr_mask)
			logits1 = tf.Print(logits1,[tf.nn.softmax(logits1)],message="logits1",summarize=100)
			logits2 = tf.Print(logits2,[tf.nn.softmax(logits2)],message="logits2",summarize=100)
			tf.summary.histogram('rQ_init',init)
			tf.summary.histogram('pointer_logits_1',logits1)
			tf.summary.histogram('pointer_logits_2',logits2)

		with tf.variable_scope("predict"):
			outer = tf.matmul(tf.expand_dims(tf.nn.softmax(logits1), axis=2),
							  tf.expand_dims(tf.nn.softmax(logits2), axis=1))
			outer = tf.matrix_band_part(outer, 0, 15)
			self.yp1 = tf.argmax(tf.reduce_max(outer, axis=2), axis=1)
			self.yp2 = tf.argmax(tf.reduce_max(outer, axis=1), axis=1)

			def condition_j(y1,y2,j,ny1,ny2,cm,cl):
				return tf.less(j, N)
			
			def batch_j(y1,y2,j,new_y1,new_y2,cm,cl):
				loop_var_i = tf.constant(0, tf.int32)
				#loop_var_i = tf.Print(loop_var_i,[loop_var_i],message="loop_var_i")
				y1, y2, j, i, new_y1, new_y2, cm, cl = tf.while_loop(condition_i,
				passage_i, loop_vars=[y1, y2, j, loop_var_i, new_y1, new_y2, cm, cl],
				shape_invariants=[tf.TensorShape([None]), tf.TensorShape([None]), j.get_shape(),
				loop_var_i.get_shape(), tf.TensorShape([N]), tf.TensorShape([N]),
				tf.TensorShape([None]), tf.TensorShape([None,None])])

				j = tf.add(j, tf.constant(1))
				return y1,y2,j,new_y1,new_y2,cm,cl
			def passage_i(y1,y2,j,i,new_y1,new_y2,cm,cl):
				def c1_1():
					indices1 = tf.reshape(j,[1, 1])
					updates1 = tf.reshape(cm[i],[1])
					shape1 = tf.reshape(N, [1])
					scatter1 = tf.scatter_nd(indices1, updates1, shape1)
					y1_ = tf.subtract(y1,scatter1)
					indices2 = tf.reshape(j,[1, 1])
					updates2 = tf.reshape(cl[j,i],[1])
					shape2 = tf.reshape(N, [1])
					scatter2 = tf.scatter_nd(indices2, updates2, shape2)
					new_y1_ = tf.add(new_y1,scatter2)
					return y1_,new_y1_
				def c2_1():
					indices1 = tf.reshape(j,[1, 1])
					updates1 = tf.reshape(y1[j],[1])
					shape1 = tf.reshape(N, [1])
					scatter1 = tf.scatter_nd(indices1, updates1, shape1)
					new_y1_ = tf.add(new_y1,scatter1)
					return y1,new_y1_
				def c1_2():
					indices1 = tf.reshape(j,[1, 1])
					updates1 = tf.reshape(cm[i],[1])
					shape1 = tf.reshape(N, [1])
					scatter1 = tf.scatter_nd(indices1, updates1, shape1)
					y2_ = tf.subtract(y2,scatter1)
					indices2 = tf.reshape(j,[1, 1])
					updates2 = tf.reshape(cl[j,i],[1])
					shape2 = tf.reshape(N, [1])
					scatter2 = tf.scatter_nd(indices2, updates2, shape2)
					new_y2_ = tf.add(new_y2,scatter2)
					return y2_,new_y2_
				def c2_2():
					indices1 = tf.reshape(j,[1, 1])
					updates1 = tf.reshape(y2[j],[1])
					shape1 = tf.reshape(N, [1])
					scatter1 = tf.scatter_nd(indices1, updates1, shape1)
					new_y2_ = tf.add(new_y2,scatter1)
					return y2,new_y2_
				#y1,new_y1 = tf.cond(cond_i_1, c1_1, c2_1)
				#y2,new_y2 = tf.cond(cond_i_2, c1_2, c2_2)
				#i = tf.Print(i,[i],message="loop_var_i")
				#j = tf.Print(j,[j],message="loop_var_j")
				y1,new_y1 = tf.cond(y1[j] > cm[i], c1_1, c2_1)
				y2,new_y2 = tf.cond(y2[j] > cm[i], c1_2, c2_2)
				i = tf.add(i, tf.constant(1))
				return y1,y2,j,i,new_y1,new_y2,cm,cl

			def condition_i(y1,y2,j,i,ny1,ny2,cm,cl):
				#self.para_count = tf.Print(self.para_count,[self.para_count[j]],message="para_count j")
				return tf.less(i, self.para_count[j])

			new_yp1 = tf.zeros([N], tf.int32)
			new_yp2 = tf.zeros([N], tf.int32)
			#cmax_c = tf.cast(cmax_c,tf.int32)
			#clen_c = tf.cast(clen_c,tf.int32)
			loop_var_j = tf.constant(0, tf.int32)
			self.yp1, self.yp2 = tf.cast(self.yp1,tf.int32), tf.cast(self.yp2,tf.int32)

			self.yp1, self.yp2, loop_var_j, new_yp1, new_yp2, cm, cl = tf.while_loop(condition_j,
				batch_j, loop_vars=[self.yp1, self.yp2, loop_var_j, new_yp1, new_yp1,\
									cmax_c, clen_c],\
				shape_invariants=[tf.TensorShape([None]), tf.TensorShape([None]), loop_var_j.get_shape(),
				tf.TensorShape([N]), tf.TensorShape([N]), tf.TensorShape([None]), tf.TensorShape([None,None])])
			
			#self.yp1 = tf.Print(self.yp1,[self.yp1],message="yp1",summarize=N)
			
			losses = tf.nn.softmax_cross_entropy_with_logits(
				logits=logits1, labels=self.y1)
			losses2 = tf.nn.softmax_cross_entropy_with_logits(
				logits=logits2, labels=self.y2)
			losses = tf.Print(losses,[losses],message="losses",summarize=20)
			losses2 = tf.Print(losses2,[losses2],message="losses2",summarize=20)
			#losses1_2 = tf.reduce_mean(losses1_2, axis=0)
			self.loss = tf.reduce_mean(losses + losses2)
			print(self.loss)
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
						vj_P = self.att_vP[:,i*400:(i+1)*400,:]
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
						vj_P = self.att_vP[:,i*400:(i+1)*400,:]
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
