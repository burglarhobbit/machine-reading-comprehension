import tensorflow as tf
import math

class R_NET:
	def random_weight(self, dim_in, dim_out, name=None, stddev=1.0):
		return tf.Variable(tf.truncated_normal([dim_in, dim_out], stddev=stddev/math.sqrt(float(dim_in))), name=name)

	def random_bias(self, dim, name=None):
		return tf.Variable(tf.truncated_normal([dim]), name=name)
	
	def random_scalar(self, name=None):
		return tf.Variable(0.0, name=name)

	def DropoutWrappedGRUCell(self, hidden_size, in_keep_prob, name=None):
		# cell = tf.contrib.rnn.GRUCell(hidden_size)
		cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, forget_bias=1.0, state_is_tuple=True)
		cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob = in_keep_prob)
		return cell

	def mat_weight_mul(self, mat, weight):
		# [batch_size, n, m] * [m, p] = [batch_size, n, p]
		mat_shape = mat.get_shape().as_list()
		weight_shape = weight.get_shape().as_list()
		assert(mat_shape[-1] == weight_shape[0])
		mat_reshape = tf.reshape(mat, [-1, mat_shape[-1]]) # [batch_size * n, m]
		mul = tf.matmul(mat_reshape, weight) # [batch_size * n, p]
		return tf.reshape(mul, [-1, mat_shape[1], weight_shape[-1]])

	def __init__(self, options):
		with tf.device('/cpu:0'):
			self.options = options
			
			# Char embeddings
			if options['char_emb']: 
				self.char_emb_mat = self.random_weight(self.options['char_vocab_size'], 
									self.options['char_emb_mat_dim'], name = 'char_emb_matrix')

			# Weights
			self.W_uQ = self.random_weight(2 * options['state_size'], options['state_size'], name='W_uQ')
			self.W_uP = self.random_weight(2 * options['state_size'], options['state_size'], name='W_uP')
			self.W_vP = self.random_weight(options['state_size'], options['state_size'], name='W_vP')
			self.W_g_QP = self.random_weight(4 * options['state_size'], 4 * options['state_size'], name='W_g_QP')
			self.W_smP1 = self.random_weight(options['state_size'], options['state_size'], name='W_smP1')
			self.W_smP2 = self.random_weight(options['state_size'], options['state_size'], name='W_smP2')
			self.W_g_SM = self.random_weight(2 * options['state_size'], 2 * options['state_size'], name='W_g_SM')
			self.W_ruQ = self.random_weight(2 * options['state_size'], 2 * options['state_size'], name='W_ruQ')
			self.W_vQ = self.random_weight(options['state_size'], 2 * options['state_size'], name='W_vQ')
			self.W_VrQ = self.random_weight(options['q_length'], options['state_size'], name='W_VrQ') # has same size as u_Q
			## changed W_hP to match v_P weights instead of h_P because of snet changes
			#self.W_hP = self.random_weight(2 * options['state_size'], options['state_size'], name='W_hP')
			self.W_hP = self.random_weight(options['state_size'], options['state_size'], name='W_hP')
			self.W_ha = self.random_weight(2 * options['state_size'], options['state_size'], name='W_ha')

			# Biases
			self.B_v_QP = self.random_bias(options['state_size'], name='B_v_QP')
			self.B_v_SM = self.random_bias(options['state_size'], name='B_v_SM')
			self.B_v_rQ = self.random_bias(2 * options['state_size'], name='B_v_rQ')
			self.B_v_ap = self.random_bias(options['state_size'], name='B_v_ap')

			# QP_match
			with tf.variable_scope('QP_match') as scope:
				self.QPmatch_cell = self.DropoutWrappedGRUCell(self.options['state_size'], self.options['in_keep_prob'])
				self.QPmatch_state = self.QPmatch_cell.zero_state(self.options['batch_size'], dtype=tf.float32)

			# Ans Ptr
			with tf.variable_scope('Ans_ptr') as scope:
				self.AnsPtr_cell = self.DropoutWrappedGRUCell(2 * self.options['state_size'], self.options['in_keep_prob'])
		
	def build_model(self):
		opts = self.options

		# placeholders
		paragraph = tf.placeholder(tf.float32, [opts['batch_size'], opts['p_length'], opts['emb_dim']])
		question = tf.placeholder(tf.float32, [opts['batch_size'], opts['q_length'], opts['emb_dim']])
		answer_si = tf.placeholder(tf.float32, [opts['batch_size'], opts['p_length']])
		answer_ei = tf.placeholder(tf.float32, [opts['batch_size'], opts['p_length']])
		if opts['char_emb']:
			paragraph_c = tf.placeholder(tf.int32, [opts['batch_size'], opts['p_length'], opts['char_max_length']])
			question_c = tf.placeholder(tf.int32, [opts['batch_size'], opts['q_length'], opts['char_max_length']])

		print('Question and Passage Encoding')
		if opts['char_emb']:
			# char embedding -> word level char embedding
			paragraph_c_emb = tf.nn.embedding_lookup(self.char_emb_mat, paragraph_c) # [batch_size, p_length, char_max_length, char_emb_dim]
			question_c_emb = tf.nn.embedding_lookup(self.char_emb_mat, question_c)
			paragraph_c_list = [tf.squeeze(w, [1]) for w in tf.split(paragraph_c_emb, opts['p_length'], axis=1)]
			question_c_list = [tf.squeeze(w, [1]) for w in tf.split(question_c_emb, opts['q_length'], axis=1)]

			c_Q = []
			c_P = []
			with tf.variable_scope('char_emb_rnn') as scope:
				char_emb_fw_cell = self.DropoutWrappedGRUCell(opts['emb_dim'], 1.0)
				char_emb_bw_cell = self.DropoutWrappedGRUCell(opts['emb_dim'], 1.0)
				for t in range(opts['q_length']):
					unstacked_q_c = tf.unstack(question_c_list[t], opts['char_max_length'], 1)
					if t>0 :
						tf.get_variable_scope().reuse_variables()
					q_c_e_outputs, q_c_e_final_fw, q_c_e_final_bw = tf.contrib.rnn.static_bidirectional_rnn(
						char_emb_fw_cell, char_emb_bw_cell, unstacked_q_c, dtype=tf.float32, scope = 'char_emb')
					c_q_t = tf.concat([q_c_e_final_fw[1], q_c_e_final_bw[1]], 1)
					c_Q.append(c_q_t)
				for t in range(opts['p_length']):
					unstacked_p_c = tf.unstack(paragraph_c_list[t], opts['char_max_length'], 1)
					p_c_e_outputs, p_c_e_final_fw, p_c_e_final_bw = tf.contrib.rnn.static_bidirectional_rnn(
						char_emb_fw_cell, char_emb_bw_cell, unstacked_p_c, dtype=tf.float32, scope = 'char_emb')
					c_p_t = tf.concat([p_c_e_final_fw[1], p_c_e_final_bw[1]], 1)
					c_P.append(c_p_t)
			c_Q = tf.stack(c_Q, 1)
			c_P = tf.stack(c_P, 1)
			print('c_Q', c_Q)
			print('c_P', c_P)
		
			# Concat e and c
			eQcQ = tf.concat([question, c_Q], 2)
			ePcP = tf.concat([paragraph, c_P], 2)
		else:
			eQcQ = question
			ePcP = paragraph

		unstacked_eQcQ = tf.unstack(eQcQ, opts['q_length'], 1)
		unstacked_ePcP = tf.unstack(ePcP, opts['p_length'], 1)
		with tf.variable_scope('encoding') as scope:
			stacked_enc_fw_cells=[ self.DropoutWrappedGRUCell(opts['state_size'], opts['in_keep_prob']) for _ in range(2)]
			stacked_enc_bw_cells=[ self.DropoutWrappedGRUCell(opts['state_size'], opts['in_keep_prob']) for _ in range(2)]
			q_enc_outputs, q_enc_final_fw, q_enc_final_bw = tf.contrib.rnn.stack_bidirectional_rnn(
									stacked_enc_fw_cells, stacked_enc_bw_cells, unstacked_eQcQ, dtype=tf.float32, scope = 'context_encoding')
			tf.get_variable_scope().reuse_variables()
			p_enc_outputs, p_enc_final_fw, p_enc_final_bw = tf.contrib.rnn.stack_bidirectional_rnn(
									stacked_enc_fw_cells, stacked_enc_bw_cells, unstacked_ePcP, dtype=tf.float32, scope = 'context_encoding')
			u_Q = tf.stack(q_enc_outputs, 1)
			u_P = tf.stack(p_enc_outputs, 1)
		u_Q = tf.nn.dropout(u_Q, opts['in_keep_prob'])
		u_P = tf.nn.dropout(u_P, opts['in_keep_prob'])
		print(u_Q)
		print(u_P)

		v_P = []
		print('Question-Passage Matching')
		for t in range(opts['p_length']):
			# Calculate c_t
			W_uQ_u_Q = self.mat_weight_mul(u_Q, self.W_uQ) # [batch_size, q_length, state_size]
			tiled_u_tP = tf.concat( [tf.reshape(u_P[:, t, :], [opts['batch_size'], 1, -1])] * opts['q_length'], 1)
			W_uP_u_tP = self.mat_weight_mul(tiled_u_tP , self.W_uP)
			
			""" Removed as not a part of snet
			if t == 0:
				tanh = tf.tanh(W_uQ_u_Q + W_uP_u_tP)
			else:
				tiled_v_t1P = tf.concat( [tf.reshape(v_P[t-1], [opts['batch_size'], 1, -1])] * opts['q_length'], 1)
				W_vP_v_t1P = self.mat_weight_mul(tiled_v_t1P, self.W_vP)
				tanh = tf.tanh(W_uQ_u_Q + W_uP_u_tP + W_vP_v_t1P)
			"""
			# added below line as a replacement of above removed portion
			tanh = tf.tanh(W_uQ_u_Q + W_uP_u_tP)
			
			s_t = tf.squeeze(self.mat_weight_mul(tanh, tf.reshape(self.B_v_QP, [-1, 1])))
			a_t = tf.nn.softmax(s_t, 1)
			tiled_a_t = tf.concat( [tf.reshape(a_t, [opts['batch_size'], -1, 1])] * 2 * opts['state_size'] , 2)
			c_t = tf.reduce_sum( tf.multiply(tiled_a_t, u_Q) , 1) # [batch_size, 2 * state_size]

			# gate
			u_tP_c_t = tf.expand_dims( tf.concat( [tf.squeeze(u_P[:, t, :]), c_t], 1), 1)
			g_t = tf.sigmoid( self.mat_weight_mul(u_tP_c_t, self.W_g_QP) )
			u_tP_c_t_star = tf.squeeze(tf.multiply(u_tP_c_t, g_t))

			# QP_match
			with tf.variable_scope("QP_match"):
				if t > 0: tf.get_variable_scope().reuse_variables()
				output, self.QPmatch_state = self.QPmatch_cell(u_tP_c_t_star, self.QPmatch_state)
				v_P.append(output)
		v_P = tf.stack(v_P, 1)
		v_P = tf.nn.dropout(v_P, opts['in_keep_prob'])
		print('v_P', v_P)

		"""
		print('Self-Matching Attention')
		SM_star = []
		for t in range(opts['p_length']):
			# Calculate s_t
			W_p1_v_P = self.mat_weight_mul(v_P, self.W_smP1) # [batch_size, p_length, state_size]
			tiled_v_tP = tf.concat( [tf.reshape(v_P[:, t, :], [opts['batch_size'], 1, -1])] * opts['p_length'], 1)
			W_p2_v_tP = self.mat_weight_mul(tiled_v_tP , self.W_smP2)
			
			tanh = tf.tanh(W_p1_v_P + W_p2_v_tP)
			s_t = tf.squeeze(self.mat_weight_mul(tanh, tf.reshape(self.B_v_SM, [-1, 1])))
			a_t = tf.nn.softmax(s_t, 1)
			tiled_a_t = tf.concat( [tf.reshape(a_t, [opts['batch_size'], -1, 1])] * opts['state_size'] , 2)
			c_t = tf.reduce_sum( tf.multiply(tiled_a_t, v_P) , 1) # [batch_size, 2 * state_size]

			# gate
			v_tP_c_t = tf.expand_dims( tf.concat( [tf.squeeze(v_P[:, t, :]), c_t], 1), 1)
			g_t = tf.sigmoid( self.mat_weight_mul(v_tP_c_t, self.W_g_SM) )
			v_tP_c_t_star = tf.squeeze(tf.multiply(v_tP_c_t, g_t))
			SM_star.append(v_tP_c_t_star)
		SM_star = tf.stack(SM_star, 1)
		unstacked_SM_star = tf.unstack(SM_star, opts['p_length'], 1)
		with tf.variable_scope('Self_match') as scope:
			SM_fw_cell = self.DropoutWrappedGRUCell(opts['state_size'], opts['in_keep_prob'])
			SM_bw_cell = self.DropoutWrappedGRUCell(opts['state_size'], opts['in_keep_prob'])
			SM_outputs, SM_final_fw, SM_final_bw = tf.contrib.rnn.static_bidirectional_rnn(SM_fw_cell, SM_bw_cell, unstacked_SM_star, dtype=tf.float32)
			h_P = tf.stack(SM_outputs, 1)
		h_P = tf.nn.dropout(h_P, opts['in_keep_prob'])
		print('h_P', h_P)
		"""

		print('Output Layer')
		# calculate r_Q
		W_ruQ_u_Q = self.mat_weight_mul(u_Q, self.W_ruQ) # [batch_size, q_length, 2 * state_size]
		W_vQ_V_rQ = tf.matmul(self.W_VrQ, self.W_vQ)
		W_vQ_V_rQ = tf.stack([W_vQ_V_rQ]*opts['batch_size'], 0) # stack -> [batch_size, state_size, state_size]
		
		tanh = tf.tanh(W_ruQ_u_Q + W_vQ_V_rQ)
		s_t = tf.squeeze(self.mat_weight_mul(tanh, tf.reshape(self.B_v_rQ, [-1, 1])))
		a_t = tf.nn.softmax(s_t, 1)
		tiled_a_t = tf.concat( [tf.reshape(a_t, [opts['batch_size'], -1, 1])] * 2 * opts['state_size'] , 2)
		r_Q = tf.reduce_sum( tf.multiply(tiled_a_t, u_Q) , 1) # [batch_size, 2 * state_size]
		r_Q = tf.nn.dropout(r_Q, opts['in_keep_prob'])
		print('r_Q', r_Q)

		# r_Q as initial state of ans ptr
		h_a = None
		p = [None for _ in range(2)]
		for t in range(2):
			### changed because of removal of self-matching attention
			#W_hP_h_P = self.mat_weight_mul(h_P, self.W_hP) # [batch_size, p_length, state_size]
			print(self.W_hP)
			W_hP_v_P = self.mat_weight_mul(v_P, self.W_hP) # [batch_size, p_length, state_size]
			
			if t == 0:
				h_t1a = r_Q
			else:
				h_t1a = h_a
			print('h_t1a', h_t1a)
			tiled_h_t1a = tf.concat( [tf.reshape(h_t1a, [opts['batch_size'], 1, -1])] * opts['p_length'], 1)
			W_ha_h_t1a = self.mat_weight_mul(tiled_h_t1a , self.W_ha)
			
			tanh = tf.tanh(W_hP_v_P + W_ha_h_t1a)
			s_t = tf.squeeze(self.mat_weight_mul(tanh, tf.reshape(self.B_v_ap, [-1, 1])))
			a_t = tf.nn.softmax(s_t, 1)
			p[t] = a_t

			## replaced the lines with appropriate snet representation. i.e v_P instead of h_P
			#tiled_a_t = tf.concat( [tf.reshape(a_t, [opts['batch_size'], -1, 1])] * 2 * opts['state_size'] , 2)
			tiled_a_t = tf.concat( [tf.reshape(a_t, [opts['batch_size'], -1, 1])] * opts['state_size'] , 2)
			#c_t = tf.reduce_sum( tf.multiply(tiled_a_t, h_P) , 1) # [batch_size, 2 * state_size]
			c_t = tf.reduce_sum( tf.multiply(tiled_a_t, v_P) , 1) # [batch_size, state_size]

			if t == 0:
				AnsPtr_state = self.AnsPtr_cell.zero_state(opts['batch_size'], dtype=tf.float32)
				h_a, _ = self.AnsPtr_cell(c_t, (AnsPtr_state, r_Q) )
				h_a = h_a[1]
				print(h_a)
		print(p)	
		p1 = p[0]
		p2 = p[1]	

		answer_si_idx = tf.cast(tf.argmax(answer_si, 1), tf.int32)
		answer_ei_idx = tf.cast(tf.argmax(answer_ei, 1), tf.int32)
		
		"""	
		ce_si = tf.nn.softmax_cross_entropy_with_logits(labels = answer_si, logits = p1)
		ce_ei = tf.nn.softmax_cross_entropy_with_logits(labels = answer_ei, logits = p2)
		print(ce_si, ce_ei)
		loss_si = tf.reduce_sum(ce_si)
		loss_ei = tf.reduce_sum(ce_ei)
		loss = loss_si + loss_ei
		"""
		
		batch_idx = tf.reshape(tf.range(0, opts['batch_size']), [-1,1])
		answer_si_re = tf.reshape(answer_si_idx, [-1,1])
		batch_idx_si = tf.concat([batch_idx, answer_si_re],1)
		answer_ei_re = tf.reshape(answer_ei_idx, [-1,1])
		batch_idx_ei = tf.concat([batch_idx, answer_ei_re],1)
		
		log_prob = tf.multiply(tf.gather_nd(p1, batch_idx_si), tf.gather_nd(p2, batch_idx_ei))
		loss = -tf.reduce_sum(tf.log(log_prob+0.0000001))
		
		# Search
		prob = []
		search_range = opts['p_length'] - opts['span_length']
		for i in range(search_range):
			for j in range(opts['span_length']):
				prob.append(tf.multiply(p1[:, i], p2[:, i+j]))
		prob = tf.stack(prob, axis = 1)
		argmax_idx = tf.argmax(prob, axis=1)
		pred_si = argmax_idx / opts['span_length']
		pred_ei = pred_si + tf.cast(tf.mod(argmax_idx , opts['span_length']), tf.float64) #tf.float64
		correct = tf.logical_and(tf.equal(tf.cast(pred_si, tf.int64), tf.cast(answer_si_idx, tf.int64)), 
								 tf.equal(tf.cast(pred_ei, tf.int64), tf.cast(answer_ei_idx, tf.int64)))
		accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

		input_tensors = {
			'p':paragraph,
			'q':question,
			'a_si':answer_si,
			'a_ei':answer_ei,
		}
		if opts['char_emb']:
			input_tensors.update({'pc': paragraph_c, 'qc': question_c})
	
		print('Model built')
		for v in tf.global_variables():
			print(v.name, v.shape)
		print('returning model')
		return input_tensors, loss, accuracy, pred_si, pred_ei

