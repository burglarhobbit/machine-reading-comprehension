import tensorflow as tf
from func import cudnn_gru, native_gru, dot_attention, summ, dropout, ptr_net, pr_attention, summ2
import sys

class Model(object):
	def __init__(self, config, batch, word_mat=None, char_mat=None, trainable=True, opt=True):
		self.config = config
		self.global_step = tf.get_variable('global_step', shape=[], dtype=tf.int32,
										   initializer=tf.constant_initializer(0), trainable=False)
		self.c, self.q, self.ch, self.qh, self.y1, self.y2, self.qa_id, self.c_pr, self.ch_pr, \
			self.passage_count, self.passage_ranking = batch.get_next()

		self.passage_count = tf.cast(self.passage_count,tf.int32)

		self.is_train = tf.get_variable(
			"is_train", shape=[], dtype=tf.bool, trainable=False)
		self.word_mat = tf.get_variable("word_mat", initializer=tf.constant(
			word_mat, dtype=tf.float32), trainable=False)
		self.char_mat = tf.get_variable(
			"char_mat", char_mat.shape, dtype=tf.float32)

		self.c_mask = tf.cast(self.c, tf.bool)
		self.q_mask = tf.cast(self.q, tf.bool)
		self.c_len = tf.reduce_sum(tf.cast(self.c_mask, tf.int32), axis=1)
		self.q_len = tf.reduce_sum(tf.cast(self.q_mask, tf.int32), axis=1)

		"""
		### this line will replace the c_len with values 8 as it is some
		# unnecessary padding from the examples which does not have
		# passages with the same number as the max number of passage in the batch
		eight_indexes = tf.not_equal(self.c_len, tf.constant(8,dtype=tf.int32))
		eight_indexes = tf.cast(eight_indexes,tf.int32)
		self.c_len = self.c_len*eight_indexes
		"""
		max_para = tf.reduce_max(self.passage_count)

		if opt:
			N, CL = config.batch_size, config.char_limit
			self.c_maxlen = tf.reduce_max(self.c_len)
			self.q_maxlen = tf.reduce_max(self.q_len)
			self.c = tf.slice(self.c, [0, 0], [N, self.c_maxlen])
			self.q = tf.slice(self.q, [0, 0], [N, self.q_maxlen])
			self.c_mask = tf.slice(self.c_mask, [0, 0], [N, self.c_maxlen])
			self.q_mask = tf.slice(self.q_mask, [0, 0], [N, self.q_maxlen])
			self.ch = tf.slice(self.ch, [0, 0, 0], [N, self.c_maxlen, CL])
			self.qh = tf.slice(self.qh, [0, 0, 0], [N, self.q_maxlen, CL])
			self.y1 = tf.slice(self.y1, [0, 0], [N, self.c_maxlen])
			self.y2 = tf.slice(self.y2, [0, 0], [N, self.c_maxlen])
			self.passage_ranking = tf.slice(self.passage_ranking, [0, 0], [N, max_para])
		else:
			self.c_maxlen, self.q_maxlen = config.para_limit, config.ques_limit

		self.ch_len = tf.reshape(tf.reduce_sum(
			tf.cast(tf.cast(self.ch, tf.bool), tf.int32), axis=2), [-1])
		self.qh_len = tf.reshape(tf.reduce_sum(
			tf.cast(tf.cast(self.qh, tf.bool), tf.int32), axis=2), [-1])

		self.rnn1 = None
		self.rnn2 = None

		self.ready()
		self.merged = tf.summary.merge_all()

		if trainable:
			self.lr = tf.get_variable(
				"lr", shape=[], dtype=tf.float32, trainable=False)
			self.opt = tf.train.AdadeltaOptimizer(
				learning_rate=self.lr, epsilon=1e-6)
			if config.with_passage_ranking:
				grads = self.opt.compute_gradients(self.ee_loss)
			else:
				grads = self.opt.compute_gradients(self.loss)
			gradients, variables = zip(*grads)
			capped_grads, _ = tf.clip_by_global_norm(
				gradients, config.grad_clip)
			self.train_op = self.opt.apply_gradients(
				zip(capped_grads, variables), global_step=self.global_step)

	def get_vp(self,i):
		config = self.config
	
		gru = cudnn_gru if config.use_cudnn else native_gru
		opt = True
		MPL = config.single_para_limit

		zero = tf.constant(0)
		i_ = tf.constant(i)
		start = i*MPL
		end = (i+1)*MPL
		c_pr = self.c_pr[:,start:end]
		ch_pr = self.ch_pr[:,start:end,:]
		
		# local masks
		c_mask = tf.cast(c_pr, tf.bool)
		q_mask = tf.cast(self.q, tf.bool)
		c_len = tf.reduce_sum(tf.cast(c_mask, tf.int32), axis=1)
		q_len = tf.reduce_sum(tf.cast(q_mask, tf.int32), axis=1)

		"""
		### this line will replace the c_len with values 8 as it is some
		# unnecessary padding from the examples which does not have
		# passages with the same number as the max number of passage in the batch
		eight_indexes = tf.not_equal(c_len, tf.constant(8,dtype=tf.int32))
		eight_indexes = tf.cast(eight_indexes,tf.int32)
		c_len = c_len*eight_indexes
		"""

		if opt:
			N, CL = config.batch_size, config.char_limit
			c_maxlen = tf.reduce_max(c_len)
			q_maxlen = tf.reduce_max(q_len)
			c_pr = tf.slice(c_pr, [0, 0], [N, c_maxlen])
			q = tf.slice(self.q, [0, 0], [N, q_maxlen])
			c_mask = tf.slice(c_mask, [0, 0], [N, c_maxlen])
			q_mask = tf.slice(q_mask, [0, 0], [N, q_maxlen])
			ch_pr = tf.slice(ch_pr, [0, 0, 0], [N, c_maxlen, CL])
			qh = tf.slice(self.qh, [0, 0, 0], [N, q_maxlen, CL])
			y1 = tf.slice(self.y1, [0, 0], [N, c_maxlen])
			y2 = tf.slice(self.y2, [0, 0], [N, c_maxlen])

			seq_mask = tf.sequence_mask(c_len, maxlen=c_maxlen)
		else:
			self.c_maxlen, self.q_maxlen = config.para_limit, config.ques_limit

		ch_len = tf.reshape(tf.reduce_sum(
			tf.cast(tf.cast(ch_pr, tf.bool), tf.int32), axis=2), [-1])
		qh_len = tf.reshape(tf.reduce_sum(
			tf.cast(tf.cast(qh, tf.bool), tf.int32), axis=2), [-1])

		N, PL, QL, CL, d, dc, dg = config.batch_size, c_maxlen, q_maxlen, config.char_limit, config.hidden, config.char_dim, config.char_hidden
		with tf.variable_scope("emb"):
			with tf.variable_scope("char"):
				ch_emb = tf.reshape(tf.nn.embedding_lookup(
					self.char_mat, ch_pr), [N * PL, CL, dc])
				qh_emb = tf.reshape(tf.nn.embedding_lookup(
					self.char_mat, qh), [N * QL, CL, dc])
				ch_emb = dropout(
					ch_emb, keep_prob=config.keep_prob, is_train=self.is_train)
				qh_emb = dropout(
					qh_emb, keep_prob=config.keep_prob, is_train=self.is_train)
				
				#self.cell_fw = tf.contrib.rnn.GRUCell(dg)
				#self.cell_bw = tf.contrib.rnn.GRUCell(dg)
				_, (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
					self.cell_fw, self.cell_bw, ch_emb, ch_len, dtype=tf.float32)
				ch_emb = tf.concat([state_fw, state_bw], axis=1)
				_, (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
					self.cell_fw, self.cell_bw, qh_emb, qh_len, dtype=tf.float32)
				qh_emb = tf.concat([state_fw, state_bw], axis=1)
				qh_emb = tf.reshape(qh_emb, [N, QL, 2 * dg])
				ch_emb = tf.reshape(ch_emb, [N, PL, 2 * dg])

			with tf.name_scope("word"):
				c_emb = tf.nn.embedding_lookup(self.word_mat, c_pr)
				q_emb = tf.nn.embedding_lookup(self.word_mat, q)

			c_emb = tf.concat([c_emb, ch_emb], axis=2)
			q_emb = tf.concat([q_emb, qh_emb], axis=2)

		with tf.variable_scope("encoding"):
			#gru1 = lambda: gru(num_layers=3, num_units=d, batch_size=N, input_size=c_emb.get_shape(
			#	).as_list()[-1], keep_prob=config.keep_prob, is_train=self.is_train)
			#self.rnn1 = tf.cond(tf.equal(i_,zero), gru1, lambda: self.rnn1)
			#c = self.rnn1(c_emb, seq_len=c_len)
			#q = self.rnn1(q_emb, seq_len=q_len)

			if i==0:
				self.rnn1 = gru(num_layers=3, num_units=d, batch_size=N, input_size=c_emb.get_shape(
				).as_list()[-1], keep_prob=config.keep_prob, is_train=self.is_train)
				self.q_enc = self.rnn1(q_emb, seq_len=q_len)
			c = self.rnn1(c_emb, seq_len=c_len)
			
		with tf.variable_scope("attention"):
			qc_att = dot_attention(c, self.q_enc, mask=q_mask, hidden=d,
				keep_prob=config.keep_prob, is_train=self.is_train, name_scope="attention_layer")
			
			#gru2 = lambda: gru(num_layers=1, num_units=d, batch_size=N, input_size=qc_att.get_shape(
			#	).as_list()[-1], keep_prob=config.keep_prob, is_train=self.is_train)
			#self.rnn2 = tf.cond(tf.equal(i_,zero), gru2, lambda: self.rnn2)
			#att = self.rnn2(qc_att, seq_len=c_len)

			if i==0:
				self.rnn2 = gru(num_layers=1, num_units=d, batch_size=N, input_size=qc_att.get_shape(
				).as_list()[-1], keep_prob=config.keep_prob, is_train=self.is_train)
			att = self.rnn2(qc_att, seq_len=c_len)
		return att, c_len, c_mask, y1, y2, seq_mask

	def ready(self):
		config = self.config
		N, PL, QL, CL, d, dc, dg = config.batch_size, self.c_maxlen, self.q_maxlen, config.char_limit, config.hidden, config.char_dim, config.char_hidden
		gru = cudnn_gru if config.use_cudnn else native_gru

		max_para = tf.reduce_max(self.passage_count)
		self.cell_fw = tf.contrib.rnn.GRUCell(dg)
		self.cell_bw = tf.contrib.rnn.GRUCell(dg)

		vp_concat = tf.zeros([N,1,300],tf.float32)
		clen_concat = tf.zeros([N,1],tf.int32)
		c_mask_concat = tf.cast(tf.zeros([N,1]),tf.bool)
		y1_concat = y2_concat = tf.zeros([N,1])
		seq_mask_concat = tf.cast(tf.zeros([N,1]),tf.bool) # maybe seq mask is = c_mask
		q = tf.zeros([N,1,1])
		for i in range(config.max_para):
			i_ = tf.constant(i)
			#print_out(i)
			def vp():
				att, c_len, c_mask, y1, y2, seq_mask = self.get_vp(i)
				
				c_len = tf.reshape(c_len,[N,1])
				att, c_len, c_mask, y1, y2, seq_mask = tf.cond(
					tf.equal(i_,tf.constant(0)),
					lambda: (att, c_len, c_mask, y1, y2, seq_mask),
					lambda: (
						tf.concat([vp_concat, att], axis=1),
						tf.concat([clen_concat, c_len], axis=1),
						tf.concat([c_mask_concat, c_mask], axis=1),
						tf.concat([y1_concat, y1], axis=1),
						tf.concat([y2_concat, y2], axis=1),
						tf.concat([seq_mask_concat, seq_mask], axis=1),
					)
				)
				return att, c_len, c_mask, y1, y2, seq_mask

			def dummy(): return vp_concat, clen_concat, c_mask_concat, y1_concat, y2_concat, seq_mask_concat

			vp_concat, clen_concat, c_mask_concat, y1_concat, y2_concat, seq_mask_concat \
				= tf.cond(i_ < max_para, vp, dummy)

		vp_mask_count = tf.reduce_sum(clen_concat, axis=1)
		
		# max count w.r.t original concatenated context (self.c_len)
		vpmccl = vp_mask_max_count_c_like = tf.reduce_max(vp_mask_count)
		# max count w.r.t concatenated vp (self.att_vP) 
		##### not used:
		vp_mask_max_count = tf.reduce_max(tf.reduce_max(clen_concat))
		
		vp_final_pad_meta = vp_mask_max_count_c_like - vp_mask_count

		# dont know why this diff happens, but it does
		diff = tf.shape(self.c_mask)[-1] - vp_mask_max_count_c_like

		vp_final_pad_seq = tf.sequence_mask(vp_final_pad_meta+diff)
		seq_mask_concat1 = tf.concat([seq_mask_concat, vp_final_pad_seq], axis=1)

		pad_length = tf.reduce_max(vp_final_pad_meta)+diff
		paddings = tf.convert_to_tensor([[0, 0], [0, pad_length], [0, 0]])
		new_vp = tf.pad(vp_concat, paddings, "CONSTANT")

		new_vp = tf.reshape(tf.boolean_mask(new_vp, seq_mask_concat1), 
			[N, vpmccl+diff, 2*config.hidden]
		)

		"""
		new_vp = tf.Print(new_vp,["vp_mask_max_count_c_like",vp_mask_max_count_c_like,
			"vp_final_pad_meta",vp_final_pad_meta,
			"vp_concat",tf.shape(vp_concat),"new_vp",tf.shape(new_vp),
			"c_mask",tf.shape(self.c_mask),"seq_mask_concat",tf.shape(seq_mask_concat),
			"clen_concat",clen_concat,"c_mask_last",self.c_mask[:,-1],
			"vp_mask_count",vp_mask_count,"c_len",self.c_len],
			summarize=N*10,message="SHORT")
		"""
		
		#self.c_mask = tf.concat([self.c_mask,vp_final_pad_seq],axis=1)
		with tf.variable_scope("pointer"):
			# r_Q:
			init = summ(self.q_enc[:, :, -2 * d:], d, mask=self.q_mask,
						keep_prob=config.ptr_keep_prob, is_train=self.is_train)

			pointer = ptr_net(batch=N, hidden=init.get_shape().as_list(
			)[-1], keep_prob=config.ptr_keep_prob, is_train=self.is_train)
			#logits1, logits2 = pointer(init, new_vp, d, self.c_mask)
			logits1, logits2 = pointer(init, new_vp, d, self.c_mask)

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
			self.loss = tf.reduce_mean(losses + losses2)
			#losses = tf.nn.softmax_cross_entropy_with_logits_v2(
			#	logits=logits1, labels=tf.stop_gradient(self.y1))
			#losses2 = tf.nn.softmax_cross_entropy_with_logits_v2(
			#	logits=logits2, labels=tf.stop_gradient(self.y2))
			#self.loss = tf.reduce_mean(losses + losses2)

		c_max = tf.reduce_max(clen_concat, axis=0)
		print(c_max)
		g_concat = tf.zeros([N,1])

		count = tf.constant(0)

		if config.with_passage_ranking:
			with tf.variable_scope("passage_ranking"):
				
				for i in range(config.max_para):
					i_ = tf.constant(i)
					def passage_ranking():
						global count
						print(c_max[i])
						if i==0:
							#vp = tf.slice(vp_concat,[0,0,0],[N,c_max[i],])
							c_max1 = tf.Print(c_max,[c_max],message="C_MAX")
							
							vp = vp_concat[:,:c_max[i],:]
							mask = seq_mask_concat[:,:c_max[i]]
							count = c_max[i]
						else:
							vp = vp_concat[:,count:count+c_max[i],:]
							mask = seq_mask_concat[:,count:count+c_max[i]]
							count += c_max[i]

						#g = pr_attention(init, vp, mask=mask, hidden=d,
						#	keep_prob=config.keep_prob, is_train=self.is_train, name_scope="rP_attention")
						#g = tf.reshape(g,[N,1])
						g = summ2(vp, init, max_para, d, mask, keep_prob=config.keep_prob,
							is_train=self.is_train, scope="summ")

						if i==0:
							return g,count
						return tf.concat([g_concat,g],axis=1),count
					def dummy():
						return g_concat,count
					g_concat,count = tf.cond(i_ < max_para, passage_ranking,dummy)

			self.losses3 = tf.nn.softmax_cross_entropy_with_logits(
						logits=g_concat, labels=self.passage_ranking)
			#self.losses3 = tf.Print(self.losses3,[self.losses3,tf.reduce_max(self.losses3),
			#	tf.reduce_max(self.pr),tf.reduce_max(gi)],message="losses3:")
			self.pr_loss = tf.reduce_mean(self.losses3)
			#self.pr_loss = tf.Print(self.pr_loss,[self.pr_loss])
			r = tf.constant(0.8)
			one_minus_r = tf.constant(0.2)
			self.ee_loss1 = tf.multiply(r,self.loss)
			self.ee_loss2 = tf.multiply(one_minus_r,self.pr_loss)
			self.ee_loss = tf.add(self.ee_loss1, self.ee_loss2)
			#self.ee_loss = tf.Print(self.ee_loss,[self.ee_loss,self.pr_loss],message="ee_loss",
			#	summarize=N*2)


	def print(self):
		pass

	def get_loss(self):
		return self.loss

	def get_global_step(self):
		return self.global_step

def print_out(s, f=None, new_line=True):
	"""Similar to print but with support to flush and output to a file."""
	if isinstance(s, bytes):
		s = s.decode("utf-8")

	if f:
		f.write(s.encode("utf-8"))
		if new_line:
			f.write(b"\n")

	# stdout
	out_s = s.encode("utf-8")
	if not isinstance(out_s, str):
		out_s = out_s.decode("utf-8")
	print(out_s, end="", file=sys.stdout)

	if new_line:
		sys.stdout.write("\n")
	sys.stdout.flush()
