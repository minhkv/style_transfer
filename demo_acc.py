import tensorflow as tf

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

logits = [[0.1, 0.5, 0.4],
          [0.8, 0.1, 0.1],
          [0.6, 0.3, 0.2]]
labels = [[0, 1, 0],
          [1, 0, 0],
          [0, 0, 1]]

acc, acc_op = tf.metrics.accuracy(labels=tf.argmax(labels, 1), 
                                  predictions=tf.argmax(logits,1))
init_g = tf.global_variables_initializer()
init_l = tf.local_variables_initializer()

sess.run(init_g)
sess.run(init_l)

print(sess.run([acc, acc_op]))
print(sess.run([acc]))