import tensorflow as tf 

x = tf.string("data.txt")

print(tf.read_file(x))
