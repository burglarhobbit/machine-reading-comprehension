import tensorflow as tf
max_para = tf.placeholder(tf.int32)
num_units = 150
inputs = tf.placeholder(tf.float32,shape=[15,8,num_units])
gru_fw = tf.contrib.cudnn_rnn.CudnnGRU(1, num_units, 
			kernel_initializer=tf.random_normal_initializer(stddev=0.1))
init_fw = tf.zeros(shape=[1, 8, num_units])

class cudnn_gru:
	def __call__(self,inputs):
		out_fw, _ = gru_fw(inputs)

class cudnn_gru2:
	def __init__(self):
		self.gru_fw = tf.contrib.cudnn_rnn.CudnnGRU(1, num_units-1, 
			kernel_initializer=tf.random_normal_initializer(stddev=0.1))
		with tf.variable_scope('CUDNN_GRU', reuse=tf.AUTO_REUSE):
			self.init_fw = tf.get_variable("init_fw",shape=[1, 8, num_units],initializer=
				tf.zeros_initializer())
			self.init_bw = tf.get_variable("init_bw",shape=[1, 8, num_units],initializer=
				tf.zeros_initializer())
	def __call__(self,inputs):
		out_fw, _ = self.gru_fw(inputs, initial_state=(self.init_fw,))

def get_output():
	gru = cudnn_gru()
	out = gru(inputs)
	return tf.constant(1)

def get_output2():
	gru = cudnn_gru2()
	out = gru(inputs)
	return tf.constant(2)

for i in range(3):
	i_ = tf.constant(i)
	out = tf.cond(i_<max_para,get_output,get_output2)
