import tensorflow as tf
from tensorflow.python.ops.rnn_cell import GRUCell

INF = 1e30
class cudnn_gru:
	def __init__(self,a,b,c,d,e,f,g):
		pass
	def __call__(self,a,b,c,d,e,f,g):
		pass

"""
gru_fw = tf.contrib.cudnn_rnn.CudnnGRU(num_layers=1, num_units=150, input_size=500)

e_ = tf.random_uniform([gru_fw.params_size()], -0.1, 0.1)
f_ = tf.random_uniform([gru_fw.params_size()], -0.1, 0.1)

#a = tf.random_normal(stddev=0.1)

class cudnn_gru:

	def __init__(self, num_layers, num_units, batch_size, input_size, keep_prob=1.0, is_train=None, scope=None):
		self.num_layers = num_layers
		self.grus = []
		self.inits = []
		self.dropout_mask = []
		for layer in range(num_layers):
			input_size_ = input_size if layer == 0 else 2 * num_units
			gru_fw = tf.contrib.cudnn_rnn.CudnnGRU(
				1, num_units, kernel_initializer=a)
			gru_bw = tf.contrib.cudnn_rnn.CudnnGRU(
				1, num_units, kernel_initializer=a)
			with tf.variable_scope('CUDNN_GRU', reuse=tf.AUTO_REUSE):
				init_fw = tf.get_variable("init_fw",shape=[1, batch_size, num_units],initializer=
					tf.zeros_initializer())
				init_bw = tf.get_variable("init_bw",shape=[1, batch_size, num_units],initializer=
					tf.zeros_initializer())
			mask_fw = dropout(tf.ones([1, batch_size, input_size_], dtype=tf.float32),
							  keep_prob=keep_prob, is_train=is_train, mode=None)
			mask_bw = dropout(tf.ones([1, batch_size, input_size_], dtype=tf.float32),
							  keep_prob=keep_prob, is_train=is_train, mode=None)
			self.grus.append((gru_fw, gru_bw, ))
			self.inits.append((init_fw, init_bw, ))
			self.dropout_mask.append((mask_fw, mask_bw, ))

	def __call__(self, inputs, seq_len, keep_prob=1.0, is_train=None, concat_layers=True):
		outputs = [tf.transpose(inputs, [1, 0, 2])]
		for layer in range(self.num_layers):
			gru_fw, gru_bw = self.grus[layer]
			init_fw, init_bw = self.inits[layer]
			mask_fw, mask_bw = self.dropout_mask[layer]
			with tf.variable_scope("fw_{}".format(layer)):
				out_fw, _ = gru_fw(
					outputs[-1] * mask_fw, initial_state=(init_fw, ))
			with tf.variable_scope("bw_{}".format(layer)):
				inputs_bw = tf.reverse_sequence(
					outputs[-1] * mask_bw, seq_lengths=seq_len, seq_dim=0, batch_dim=1)
				out_bw, _ = gru_bw(inputs_bw, initial_state=(init_bw, ))
				out_bw = tf.reverse_sequence(
					out_bw, seq_lengths=seq_len, seq_dim=0, batch_dim=1)
			outputs.append(tf.concat([out_fw, out_bw], axis=2))
		if concat_layers:
			res = tf.concat(outputs[1:], axis=2)
		else:
			res = outputs[-1]
		res = tf.transpose(res, [1, 0, 2])
		return res

class cudnn_gru:

	def __init__(self, num_layers, num_units, batch_size, input_size, keep_prob=1.0, is_train=None, scope=None):
		self.num_layers = num_layers
		self.grus = []
		self.params = []
		self.inits = []
		self.dropout_mask = []
		for layer in range(num_layers):
			input_size_ = input_size if layer == 0 else 2 * num_units
			gru_fw = tf.contrib.cudnn_rnn.CudnnGRU(
				num_layers=1, num_units=num_units, input_size=input_size_)
			gru_bw = tf.contrib.cudnn_rnn.CudnnGRU(
				num_layers=1, num_units=num_units, input_size=input_size_)
			
			#param_fw = tf.Variable(tf.random_uniform(
			#	[gru_fw.params_size()], -0.1, 0.1), validate_shape=False)
			#param_bw = tf.Variable(tf.random_uniform(
			#	[gru_bw.params_size()], -0.1, 0.1), validate_shape=False)
			#init_fw = tf.Variable(tf.zeros([1, batch_size, num_units]))
			#init_bw = tf.Variable(tf.zeros([1, batch_size, num_units]))
			
			# initializer=tf.random_normal_initializer(stddev=0.1)
			#e = tf.constant_initializer(e_)
			#f = tf.constant_initializer(f_)
			with tf.variable_scope('CUDNN_GRU', reuse=tf.AUTO_REUSE):
				param_fw = tf.get_variable("param_fw",initializer=e_,validate_shape=False)
				param_bw = tf.get_variable("param_bw",initializer=f_,validate_shape=False)

				init_fw = tf.get_variable("init_fw",shape=[1, batch_size, num_units],initializer=
					tf.zeros_initializer())
				init_bw = tf.get_variable("init_bw",shape=[1, batch_size, num_units],initializer=
					tf.zeros_initializer())
			mask_fw = dropout(tf.ones([1, batch_size, input_size_], dtype=tf.float32),
							  keep_prob=keep_prob, is_train=is_train, mode=None)
			mask_bw = dropout(tf.ones([1, batch_size, input_size_], dtype=tf.float32),
							  keep_prob=keep_prob, is_train=is_train, mode=None)
			self.grus.append((gru_fw, gru_bw, ))
			self.params.append((param_fw, param_bw, ))
			self.inits.append((init_fw, init_bw, ))
			self.dropout_mask.append((mask_fw, mask_bw, ))

	def __call__(self, inputs, seq_len, keep_prob=1.0, is_train=None, concat_layers=True):
		outputs = [tf.transpose(inputs, [1, 0, 2])]
		for layer in range(self.num_layers):
			gru_fw, gru_bw = self.grus[layer]
			param_fw, param_bw = self.params[layer]
			init_fw, init_bw = self.inits[layer]
			mask_fw, mask_bw = self.dropout_mask[layer]
			with tf.variable_scope("fw"):
				out_fw, _ = gru_fw(outputs[-1] * mask_fw, init_fw, param_fw)
			with tf.variable_scope("bw"):
				inputs_bw = tf.reverse_sequence(
					outputs[-1] * mask_bw, seq_lengths=seq_len, seq_dim=0, batch_dim=1)
				out_bw, _ = gru_bw(inputs_bw, init_bw, param_bw)
				out_bw = tf.reverse_sequence(
					out_bw, seq_lengths=seq_len, seq_dim=0, batch_dim=1)
			outputs.append(tf.concat([out_fw, out_bw], axis=2))
		if concat_layers:
			res = tf.concat(outputs[1:], axis=2)
		else:
			res = outputs[-1]
		res = tf.transpose(res, [1, 0, 2])
		return res
"""
class native_gru:

	def __init__(self, num_layers, num_units, batch_size, input_size, keep_prob=1.0, is_train=None, scope="native_gru"):
		self.num_layers = num_layers
		self.grus = []
		self.inits = []
		self.dropout_mask = []
		self.scope = scope
		for layer in range(num_layers):

			#device_str = get_device_str(device_id)

			input_size_ = input_size if layer == 0 else 2 * num_units
			gru_fw = tf.contrib.rnn.GRUCell(num_units)
			gru_bw = tf.contrib.rnn.GRUCell(num_units)

			#gru_bw = tf.contrib.rnn.DeviceWrapper(gru_fw, device_str)
			#ru_bw = tf.contrib.rnn.DeviceWrapper(gru_bw, device_str)

			with tf.variable_scope('CUDNN_GRU', reuse=tf.AUTO_REUSE):
				init_fw = tf.get_variable("init_fw",shape=[batch_size, num_units],initializer=
					tf.zeros_initializer())
				init_bw = tf.get_variable("init_bw",shape=[batch_size, num_units],initializer=
					tf.zeros_initializer())
			mask_fw = dropout(tf.ones([batch_size, 1, input_size_], dtype=tf.float32),
							  keep_prob=keep_prob, is_train=is_train, mode=None)
			mask_bw = dropout(tf.ones([batch_size, 1, input_size_], dtype=tf.float32),
							  keep_prob=keep_prob, is_train=is_train, mode=None)
			self.grus.append((gru_fw, gru_bw, ))
			self.inits.append((init_fw, init_bw, ))
			self.dropout_mask.append((mask_fw, mask_bw, ))

	def __call__(self, inputs, seq_len, keep_prob=1.0, is_train=None, concat_layers=True):
		outputs = [inputs]
		with tf.variable_scope(self.scope):
			for layer in range(self.num_layers):
				gru_fw, gru_bw = self.grus[layer]
				init_fw, init_bw = self.inits[layer]
				mask_fw, mask_bw = self.dropout_mask[layer]
				with tf.variable_scope("fw_{}".format(layer)):
					out_fw, _ = tf.nn.dynamic_rnn(
						gru_fw, outputs[-1] * mask_fw, seq_len, initial_state=init_fw, dtype=tf.float32)
				with tf.variable_scope("bw_{}".format(layer)):
					inputs_bw = tf.reverse_sequence(
						outputs[-1] * mask_bw, seq_lengths=seq_len, seq_dim=1, batch_dim=0)
					out_bw, _ = tf.nn.dynamic_rnn(
						gru_fw, inputs_bw, seq_len, initial_state=init_bw, dtype=tf.float32)
					out_bw = tf.reverse_sequence(
						out_bw, seq_lengths=seq_len, seq_dim=1, batch_dim=0)
				outputs.append(tf.concat([out_fw, out_bw], axis=2))
		if concat_layers:
			res = tf.concat(outputs[1:], axis=2)
		else:
			res = outputs[-1]
		return res

def get_device_str(device_id):
	"""Return a device string for multi-GPU setup."""
	device_str_output = "/gpu:%d" % (device_id)
	return device_str_output	

class ptr_net:
	def __init__(self, batch, hidden, keep_prob=1.0, is_train=None, scope="ptr_net"):
		self.gru = tf.contrib.rnn.GRUCell(hidden)
		self.batch = batch
		self.scope = scope
		self.keep_prob = keep_prob
		self.is_train = is_train
		self.dropout_mask = dropout(tf.ones(
			[batch, hidden], dtype=tf.float32), keep_prob=keep_prob, is_train=is_train)

	def __call__(self, init, match, d, mask):
		with tf.variable_scope(self.scope):
			d_match = dropout(match, keep_prob=self.keep_prob,
							  is_train=self.is_train)
			inp, logits1 = pointer(d_match, init * self.dropout_mask, d, mask, name_scope="ptr_net_start")
			d_inp = dropout(inp, keep_prob=self.keep_prob,
							is_train=self.is_train)
			_, state = self.gru(d_inp, init)
			tf.get_variable_scope().reuse_variables()
			_, logits2 = pointer(d_match, state * self.dropout_mask, d, mask, name_scope="ptr_net_end")
			return logits1, logits2

def dropout(args, keep_prob, is_train, mode="recurrent"):
	if keep_prob < 1.0:
		noise_shape = None
		scale = 1.0
		shape = tf.shape(args)
		if mode == "embedding":
			noise_shape = [shape[0], 1]
			scale = keep_prob
		if mode == "recurrent" and len(args.get_shape().as_list()) == 3:
			noise_shape = [shape[0], 1, shape[-1]]
		args = tf.cond(is_train, lambda: tf.nn.dropout(
			args, keep_prob, noise_shape=noise_shape) * scale, lambda: args)
	return args

def softmax_mask(val, mask):
	return -INF * (1 - tf.cast(mask, tf.float32)) + val

def pointer(inputs, state, hidden, mask, scope="pointer", name_scope="pointer_layer"):
	with tf.name_scope(name_scope):
		with tf.variable_scope(scope):
			u = tf.concat([tf.tile(tf.expand_dims(state, axis=1), [
				1, tf.shape(inputs)[1], 1]), inputs], axis=2)
			s0 = tf.nn.tanh(dense(u, hidden, use_bias=False, scope="s0", name_scope="s0_layer"))
			s = dense(s0, 1, use_bias=False, scope="s", name_scope="s_layer")
			s1 = softmax_mask(tf.squeeze(s, [2]), mask)
			a = tf.expand_dims(tf.nn.softmax(s1), axis=2)
			res = tf.reduce_sum(a * inputs, axis=1)
			return res, s1


def summ(memory, hidden, mask, keep_prob=1.0, is_train=None, scope="summ"):
	with tf.variable_scope(scope):

		d_memory = dropout(memory, keep_prob=keep_prob, is_train=is_train)
		#d_memory = tf.cond(is_train,
		#	lambda: tf.nn.dropout(memory,keep_prob=keep_prob)*keep_prob,
		#	lambda: d_memory
		#)

		s0 = tf.nn.tanh(dense(d_memory, hidden, scope="s0", name_scope="summ_layer_s0"))
		s = dense(s0, 1, use_bias=False, scope="s", name_scope="summ_layer_s")
		s1 = softmax_mask(tf.squeeze(s, [2]), mask)
		a = tf.expand_dims(tf.nn.softmax(s1), axis=2)
		res = tf.reduce_sum(a * memory, axis=1)
		return res

def dot_attention(inputs, memory, mask, hidden, name_scope,
				  keep_prob=1.0, is_train=None, scope="dot_attention"):
	with tf.name_scope(name_scope):
		with tf.variable_scope(scope):

			d_inputs = dropout(inputs, keep_prob=keep_prob, is_train=is_train)
			d_memory = dropout(memory, keep_prob=keep_prob, is_train=is_train)
			JX = tf.shape(inputs)[1]

			with tf.variable_scope("attention"):
				inputs_ = tf.nn.relu(
					dense(d_inputs, hidden, use_bias=False, scope="inputs", name_scope="input_layer"))
				memory_ = tf.nn.relu(
					dense(d_memory, hidden, use_bias=False, scope="memory", name_scope="memory_layer"))
				outputs = tf.matmul(inputs_, tf.transpose(
					memory_, [0, 2, 1])) / (hidden ** 0.5)
				mask = tf.tile(tf.expand_dims(mask, axis=1), [1, JX, 1])
				logits = tf.nn.softmax(softmax_mask(outputs, mask))
				outputs = tf.matmul(logits, memory)
				res = tf.concat([inputs, outputs], axis=2)

			with tf.variable_scope("gate"):
				dim = res.get_shape().as_list()[-1]
				d_res = dropout(res, keep_prob=keep_prob, is_train=is_train)
				gate = tf.nn.sigmoid(dense(d_res, dim, use_bias=False, name_scope="gate_layer"))
				return res * gate

def dense(inputs, hidden, name_scope, use_bias=True, scope="dense"):
	with tf.name_scope(name_scope):
		with tf.variable_scope(scope,reuse=tf.AUTO_REUSE):
			shape = tf.shape(inputs)
			dim = inputs.get_shape().as_list()[-1]
			out_shape = [shape[idx] for idx in range(
				len(inputs.get_shape().as_list()) - 1)] + [hidden]
			flat_inputs = tf.reshape(inputs, [-1, dim])
			with tf.name_scope('weights'):
				W = tf.get_variable("W", [dim, hidden])
				#variable_summaries(W)
			res = tf.matmul(flat_inputs, W)
			if use_bias:
				b = tf.get_variable(
					"b", [hidden], initializer=tf.constant_initializer(0.))
				res = tf.nn.bias_add(res, b)
			res = tf.reshape(res, out_shape)
			return res

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