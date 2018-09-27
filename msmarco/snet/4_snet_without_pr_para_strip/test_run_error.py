import tensorflow as tf
gru_fw = tf.contrib.cudnn_rnn.CudnnGRU(num_layers=1, num_units=75, input_size=25)
e = tf.random_uniform([x], -0.1, 0.1)
#e = tf.placeholder(size=)
#param_fw = tf.get_variable("abcd",initializer=e, validate_shape=False) #working
i = tf.constant(0)
def func():
	#tf.get_variable_scope().reuse_variables()
	gru_fw = tf.contrib.cudnn_rnn.CudnnGRU(num_layers=1, num_units=75, input_size=25)
	# original line: commented out and working if not under a control flow mechanism
	# param_fw = tf.Variable(tf.random_uniform([gru_fw.params_size()], -0.1, 0.1), validate_shape=False)
	# converted line
	#param_fw = tf.get_variable("abcd",initializer=e, validate_shape=False)
	param_fw = tf.get_variable("abcd",initializer=e,validate_shape=False)
	return param_fw

def func2():
	### repeat the same thing from func1
	return tf.constant(1,dtype=tf.float32)

result = tf.cond(tf.equal(i, tf.constant(0)),func,func2)
result = tf.cond(tf.equal(i, tf.constant(0)),func,func2)