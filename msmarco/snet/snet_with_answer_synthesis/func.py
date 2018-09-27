import tensorflow as tf
from tensorflow.python.ops.rnn_cell import GRUCell
from tensorflow.python.layers import core as layers_core

INF = 1e30


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
			param_fw = tf.Variable(tf.random_uniform(
				[gru_fw.params_size()], -0.1, 0.1), validate_shape=False)
			param_bw = tf.Variable(tf.random_uniform(
				[gru_bw.params_size()], -0.1, 0.1), validate_shape=False)
			init_fw = tf.Variable(tf.zeros([1, batch_size, num_units]))
			init_bw = tf.Variable(tf.zeros([1, batch_size, num_units]))
			mask_fw = dropout(tf.ones([1, batch_size, input_size_], dtype=tf.float32),
							  keep_prob=keep_prob, is_train=is_train, mode=None)
			mask_bw = dropout(tf.ones([1, batch_size, input_size_], dtype=tf.float32),
							  keep_prob=keep_prob, is_train=is_train, mode=None)
			self.grus.append((gru_fw, gru_bw,))
			self.params.append((param_fw, param_bw,))
			self.inits.append((init_fw, init_bw,))
			self.dropout_mask.append((mask_fw, mask_bw,))

	def __call__(self, inputs, seq_len, keep_prob=1.0, is_train=None, concat_layers=True):
		outputs = [tf.transpose(inputs, [1, 0, 2])]
		bw_final_state = []
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
				out_bw, out_bw_final_state = gru_bw(inputs_bw, init_bw, param_bw)
				out_bw = tf.reverse_sequence(
					out_bw, seq_lengths=seq_len, seq_dim=0, batch_dim=1)
			outputs.append(tf.concat([out_fw, out_bw], axis=2))

			bw_final_state.append(out_bw_final_state)
		if concat_layers:
			res = tf.concat(outputs[1:], axis=2)
			bw_final_state = tf.concat(bw_final_state, axis=1)
		else:
			res = outputs[-1]
		# res = tf.transpose(res, [1, 0, 2])
		return res, bw_final_state


class native_gru:

	def __init__(self, num_layers, num_units, batch_size, input_size, keep_prob=1.0, is_train=None, scope="native_gru"):
		self.num_layers = num_layers
		self.grus = []
		self.inits = []
		self.dropout_mask = []
		self.scope = scope
		for layer in range(num_layers):
			input_size_ = input_size if layer == 0 else 2 * num_units
			gru_fw = tf.contrib.rnn.GRUCell(num_units)
			gru_bw = tf.contrib.rnn.GRUCell(num_units)
			init_fw = tf.Variable(tf.zeros([batch_size, num_units]))
			init_bw = tf.Variable(tf.zeros([batch_size, num_units]))
			mask_fw = dropout(tf.ones([batch_size, 1, input_size_], dtype=tf.float32),
							  keep_prob=keep_prob, is_train=is_train, mode=None)
			mask_bw = dropout(tf.ones([batch_size, 1, input_size_], dtype=tf.float32),
							  keep_prob=keep_prob, is_train=is_train, mode=None)
			self.grus.append((gru_fw, gru_bw,))
			self.inits.append((init_fw, init_bw,))
			self.dropout_mask.append((mask_fw, mask_bw,))

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
						gru_bw, inputs_bw, seq_len, initial_state=init_bw, dtype=tf.float32)
					out_bw = tf.reverse_sequence(
						out_bw, seq_lengths=seq_len, seq_dim=1, batch_dim=0)
				outputs.append(tf.concat([out_fw, out_bw], axis=2))
		if concat_layers:
			res = tf.concat(outputs[1:], axis=2)
		else:
			res = outputs[-1]
		return res


class pr_attention:
	def __init__(self, batch, hidden, keep_prob=1.0, is_train=None, scope="pr_attention",
				 name_scope="pr_attention_layer"):
		self.batch = batch
		self.scope = scope
		self.name_scope = name_scope
		self.keep_prob = keep_prob
		self.is_train = is_train
		self.dropout_mask = dropout(tf.ones(
			[batch, hidden], dtype=tf.float32), keep_prob=keep_prob, is_train=is_train)

	def __call__(self, init, match, d, mask):
		with tf.name_scope(self.name_scope):
			with tf.variable_scope(self.scope):
				d_match = dropout(match, keep_prob=self.keep_prob,
								  is_train=self.is_train)
				inp, logits1 = pointer(d_match, init * self.dropout_mask, d, mask, "pr_pointer")
				return inp


class ptr_net:
	def __init__(self, batch, hidden, keep_prob=1.0, is_train=None, scope="ptr_net"):
		self.gru = GRUCell(hidden)
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
			inp, logits1 = pointer(d_match, init * self.dropout_mask, d, mask,
								   name_scope="ptr_net_start")
			d_inp = dropout(inp, keep_prob=self.keep_prob,
							is_train=self.is_train)
			_, state = self.gru(d_inp, init)
			tf.get_variable_scope().reuse_variables()
			_, logits2 = pointer(d_match, state * self.dropout_mask, d, mask,
								 name_scope="ptr_net_end")
			return logits1, logits2


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


def summ(memory, hidden, mask, keep_prob=1.0, is_train=None, scope="summ"):
	with tf.variable_scope(scope):
		d_memory = dropout(memory, keep_prob=keep_prob, is_train=is_train)
		s0 = tf.nn.tanh(dense(d_memory, hidden, scope="s0", name_scope="summ_layer_s0"))
		s = dense(s0, 1, use_bias=False, scope="s", name_scope="summ_layer_s0")
		s1 = softmax_mask(tf.squeeze(s, [2]), mask)
		a = tf.expand_dims(tf.nn.softmax(s1), axis=2)
		res = tf.reduce_sum(a * memory, axis=1)
		return res


def dot_attention(inputs, memory, mask, hidden, name_scope, keep_prob=1.0, is_train=None, scope="dot_attention"):
	with tf.name_scope(scope):
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
				# logits = attention 'a'
				logits = tf.nn.softmax(softmax_mask(outputs, mask))
				outputs = tf.matmul(logits, memory)
				res = tf.concat([inputs, outputs], axis=2)

			with tf.variable_scope("gate"):
				dim = res.get_shape().as_list()[-1]
				d_res = dropout(res, keep_prob=keep_prob, is_train=is_train)
				gate = tf.nn.sigmoid(dense(d_res, dim, use_bias=False, name_scope="gate_layer"))
				return res * gate


def dense(inputs, hidden, name_scope="dense", use_bias=True, scope="dense"):
	with tf.name_scope(name_scope):
		with tf.variable_scope(scope):
			shape = tf.shape(inputs)
			dim = inputs.get_shape().as_list()[-1]
			out_shape = [shape[idx] for idx in range(
				len(inputs.get_shape().as_list()) - 1)] + [hidden]
			flat_inputs = tf.reshape(inputs, [-1, dim])
			W = tf.get_variable("W", [dim, hidden])
			res = tf.matmul(flat_inputs, W)
			if use_bias:
				b = tf.get_variable(
					"b", [hidden], initializer=tf.constant_initializer(0.))
				res = tf.nn.bias_add(res, b)
			res = tf.reshape(res, out_shape)
			return res


def _build_decoder(encoder_outputs, encoder_state, hparams, is_train, source_sequence_length,
				   target_sequence_length, target_input, embedding_decoder):
	# Projection
	with tf.variable_scope("build_network"):
		with tf.variable_scope("decoder/output_projection"):
			output_layer = layers_core.Dense(
				hparams.vocab_size, use_bias=False, name="output_projection")
	"""Build and run a RNN decoder with a final projection layer.
	Args:
	  encoder_outputs: The outputs of encoder for every time step.
	  encoder_state: The final state of the encoder.
	  hparams: The Hyperparameters configurations.
	Returns:
	  A tuple of final logits and final decoder state:
		logits: size [time, batch_size, vocab_size] when time_major=True.
	"""

	sos_id = tf.cast(hparams.sos_id, tf.int32)
	eos_id = tf.cast(hparams.eos_id, tf.int32)
	"""
	iterator = self.iterator
	"""
	# maximum_iteration: The maximum decoding steps.
	# maximum_iterations = self._get_infer_maximum_iterations( hparams, iterator.source_sequence_length)
	time_major = True
	## Decoder.
	with tf.variable_scope("decoder") as decoder_scope:
		cell, decoder_initial_state = _build_decoder_cell(
			hparams, encoder_outputs, encoder_state,
			source_sequence_length)

		## Train or eval
		if is_train:
			# decoder_emp_inp: [max_time, batch_size, num_units]
			target_input = target_input  # context + question idxs
			if time_major:
				target_input = tf.transpose(target_input)
			decoder_emb_inp = tf.nn.embedding_lookup(
				embedding_decoder, target_input)

			# Helper
			helper = tf.contrib.seq2seq.TrainingHelper(
				decoder_emb_inp, target_sequence_length,  # Answer length
				time_major=time_major)

			# Decoder
			my_decoder = tf.contrib.seq2seq.BasicDecoder(
				cell,
				helper,
				decoder_initial_state, )

			# Dynamic decoding
			outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
				my_decoder,
				output_time_major=time_major,
				swap_memory=True,
				scope=decoder_scope)
			sample_id = outputs.sample_id

			# sample_id = outputs.sample_id

			# Note: there's a subtle difference here between train and inference.
			# We could have set output_layer when create my_decoder
			#   and shared more code between train and inference.
			# We chose to apply the output_layer to all timesteps for speed:
			#   10% improvements for small models & 20% for larger ones.
			# If memory is a concern, we should apply output_layer per timestep.
			logits = output_layer(outputs.rnn_output)

		## Inference
		else:
			beam_width = hparams.beam_width
			length_penalty_weight = hparams.length_penalty_weight
			start_tokens = tf.fill([hparams.batch_size], sos_id)
			end_token = eos_id

			if beam_width > 0:
				my_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
					cell=cell,
					embedding=embedding_decoder,
					start_tokens=start_tokens,
					end_token=end_token,
					initial_state=decoder_initial_state,
					beam_width=beam_width,
					output_layer=output_layer,
					length_penalty_weight=length_penalty_weight)
			else:
				# Helper
				sampling_temperature = hparams.sampling_temperature
				if sampling_temperature > 0.0:
					helper = tf.contrib.seq2seq.SampleEmbeddingHelper(
						embedding_decoder, start_tokens, end_token,
						softmax_temperature=sampling_temperature,
						seed=hparams.random_seed)
				else:
					helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
						embedding_decoder, start_tokens, end_token)

				# Decoder
				my_decoder = tf.contrib.seq2seq.BasicDecoder(
					cell,
					helper,
					decoder_initial_state,
					output_layer=output_layer  # applied per timestep
				)

		# Dynamic decoding
		outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
			my_decoder,
			maximum_iterations=None,
			output_time_major=time_major,
			swap_memory=True,
			scope=decoder_scope)

		if beam_width > 0:
			logits = tf.no_op()
			sample_id = outputs.predicted_ids
		else:
			logits = outputs.rnn_output
			sample_id = outputs.sample_id

	return logits, sample_id, final_context_state


def _build_decoder_cell(hparams, encoder_outputs, encoder_state, source_sequence_length,
						is_train):
	"""Build a RNN cell with attention mechanism that can be used by decoder."""
	keep_prob = hparams.keep_prob
	attention_option = hparams.attention
	attention_architecture = hparams.attention_architecture

	if attention_architecture != "standard":
		raise ValueError(
			"Unknown attention architecture %s" % attention_architecture)

	num_units = hparams.hidden  # hparams.num_units
	num_layers = 3  # self.num_decoder_layers
	num_residual_layers = 0  # self.num_decoder_residual_layers
	beam_width = hparams.beam_width

	dtype = tf.float32

	time_major = True
	# Ensure memory is batch-major
	if time_major:
		memory = tf.transpose(encoder_outputs, [1, 0, 2])
	else:
		memory = encoder_outputs

	if is_train and beam_width > 0:
		memory = tf.contrib.seq2seq.tile_batch(
			memory, multiplier=beam_width)
		source_sequence_length = tf.contrib.seq2seq.tile_batch(
			source_sequence_length, multiplier=beam_width)
		encoder_state = tf.contrib.seq2seq.tile_batch(
			encoder_state, multiplier=beam_width)
		batch_size = hparams.batch_size * beam_width
	else:
		batch_size = hparams.batch_size

	attention_mechanism = create_attention_mechanism(
		attention_option, num_units, memory, source_sequence_length)

	cell = create_rnn_cell(
		num_units=num_units,
		num_layers=num_layers,
		keep_prob=keep_prob,
		is_train=is_train)

	# Only generate alignment in greedy INFER mode.
	alignment_history = False
	"""(self.mode == tf.contrib.learn.ModeKeys.INFER and
						# beam_width == 0)
	"""
	cell = tf.contrib.seq2seq.AttentionWrapper(
		cell,
		attention_mechanism,
		attention_layer_size=num_units,
		alignment_history=alignment_history,
		output_attention=hparams.output_attention,
		name="attention")

	# TODO(thangluong): do we need num_layers, num_gpus?
	# cell = tf.contrib.rnn.DeviceWrapper(cell,
	#									model_helper.get_device_str(
	#									num_layers - 1, self.num_gpus))

	if hparams.pass_hidden_state:
		decoder_initial_state = cell.zero_state(batch_size, dtype).clone(
			cell_state=encoder_state)
	else:
		decoder_initial_state = cell.zero_state(batch_size, dtype)

	return cell, decoder_initial_state


def create_attention_mechanism(attention_option, num_units, memory,
							   source_sequence_length):
	"""Create attention mechanism based on the attention_option."""

	# Mechanism
	if attention_option == "luong":
		attention_mechanism = tf.contrib.seq2seq.LuongAttention(
			num_units, memory, memory_sequence_length=source_sequence_length)
	elif attention_option == "scaled_luong":
		attention_mechanism = tf.contrib.seq2seq.LuongAttention(
			num_units, memory, memory_sequence_length=source_sequence_length, scale=True)
	elif attention_option == "bahdanau":
		attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
			num_units, memory, memory_sequence_length=source_sequence_length)
	elif attention_option == "normed_bahdanau":
		attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
			num_units,
			memory,
			memory_sequence_length=source_sequence_length,
			normalize=True)
	else:
		raise ValueError("Unknown attention option %s" % attention_option)

	return attention_mechanism


def create_rnn_cell(num_units, num_layers, keep_prob, is_train):
	"""Create multi-layer RNN cell.
	Args:
	unit_type: string representing the unit type, i.e. "lstm".
	num_units: the depth of each unit.
	num_layers: number of cells.
	num_residual_layers: Number of residual layers from top to bottom. For
	  example, if `num_layers=4` and `num_residual_layers=2`, the last 2 RNN
	  cells in the returned list will be wrapped with `ResidualWrapper`.
	forget_bias: the initial forget bias of the RNNCell(s).
	dropout: floating point value between 0.0 and 1.0:
	  the probability of dropout.  this is ignored if `mode != TRAIN`.
	mode: either tf.contrib.learn.TRAIN/EVAL/INFER
	num_gpus: The number of gpus to use when performing round-robin
	  placement of layers.
	base_gpu: The gpu device id to use for the first RNN cell in the
	  returned list. The i-th RNN cell will use `(base_gpu + i) % num_gpus`
	  as its device id.
	single_cell_fn: allow for adding customized cell.
	  When not specified, we default to model_helper._single_cell
	Returns:
	An `RNNCell` instance.
	"""
	cell_list = _cell_list(num_units=num_units,
						   num_layers=num_layers,
						   keep_prob=keep_prob,
						   is_train=is_train,
						   )

	if len(cell_list) == 1:  # Single layer.
		return cell_list[0]
	else:  # Multi layers
		return tf.contrib.rnn.MultiRNNCell(cell_list)


def _cell_list(num_units, num_layers, keep_prob, is_train):
	"""Create a list of RNN cells."""
	"""
	if not single_cell_fn:
		single_cell_fn = _single_cell
	"""
	# Multi-GPU
	cell_list = []
	for i in range(num_layers):
		# utils.print_out("  cell %d" % i, new_line=False)
		single_cell = tf.contrib.rnn.GRUCell(num_units * 2)
		single_cell = dropout(single_cell, keep_prob=keep_prob, is_train=is_train)

		cell_list.append(single_cell)

	return cell_list


"""
def _single_cell(unit_type, num_units, forget_bias, dropout, mode,
				 residual_connection=False, device_str=None, residual_fn=None):
	""Create an instance of a single RNN cell.""
	# dropout (= 1 - keep_prob) is set to 0 during eval and infer
	dropout = dropout if mode == tf.contrib.learn.ModeKeys.TRAIN else 0.0

	# Cell Type
	if unit_type == "lstm":
		utils.print_out("  LSTM, forget_bias=%g" % forget_bias, new_line=False)
		single_cell = tf.contrib.rnn.BasicLSTMCell(
			num_units,
			forget_bias=forget_bias)
	elif unit_type == "gru":
		utils.print_out("  GRU", new_line=False)
		single_cell = tf.contrib.rnn.GRUCell(num_units)
	elif unit_type == "layer_norm_lstm":
		utils.print_out("  Layer Normalized LSTM, forget_bias=%g" % forget_bias,
						new_line=False)
		single_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(
			num_units,
			forget_bias=forget_bias,
			layer_norm=True)

	# Dropout (= 1 - keep_prob)
	if dropout > 0.0:
		single_cell = tf.contrib.rnn.DropoutWrapper(
			cell=single_cell, input_keep_prob=(1.0 - dropout))
		utils.print_out("  %s, dropout=%g " %(type(single_cell).__name__, dropout),
						new_line=False)

	
	# Residual
	if residual_connection:
		single_cell = tf.contrib.rnn.ResidualWrapper(
			single_cell, residual_fn=residual_fn)
		utils.print_out("  %s" % type(single_cell).__name__, new_line=False)

	# Device Wrapper
	if device_str:
		single_cell = tf.contrib.rnn.DeviceWrapper(single_cell, device_str)
		utils.print_out("  %s, device=%s" %
						(type(single_cell).__name__, device_str), new_line=False)
	
	return single_cell

"""
