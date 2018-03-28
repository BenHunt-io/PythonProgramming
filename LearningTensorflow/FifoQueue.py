import tensorflow as tf 

q = tf.FIFOQueue(3, "float")
print(q.enqueue([1,2,3]))

print(q.size())
print(q.dequeue())
