import tensorflow as tf

#ignore debugging stuff
tf.logging.set_verbosity(tf.logging.ERROR)

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
print(node1,node2)

#At this points we have nodes in our graph, but not going to execute till we attach them to a session.
#This is why it doesn't show the value is 3 and 4 yet.


#Output of this shows various GPU's and info about it. It takes advantages of these in it's own scheduling algo's
sess = tf.Session()

#Now we have a session and can run the various nodes and see the output.
#We have a context , the context is attached to a computational device (GPU), We are able to evaluate the def. graph.
print(sess.run([node1,node2]))


node3 = tf.add(node1,node2) # same as sess.run(node1 + node2)

print(sess.run(node3))


#Placeholders...

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

adder_node = a + b

#Pass in the dictionary the assigns the values
print(sess.run(adder_node, {a:3, b:4.5}))

#Can also be different shaped tensors..
print(sess.run(adder_node, {a:[1,3], b:[2,4]}))


add_and_triple = adder_node * 3



#------------------------------------
#Use tensorflow variables because these will get updated (as we train), tf placeholders just get reset each
#time we pass through the graph.

W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)
x = tf.placeholder(tf.float32)


#AKA  Y = mx + b 
linear_model = W * x + b # x is input, W is weight, b is bias, both weights and bias will update, inputs is like argument

#Not sure why we have to, maybe because it's like objects in Java that have to get initialized.
init = tf.global_variables_initializer() #Global vars have to be initialized, init another node needs to be run in sess
sess.run(init)


print(sess.run(linear_model, {x:[1,2,3,4]}))

y = tf.placeholder(tf.float32) #Labels

#To train we need a loss function
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)

print(sess.run(loss,{x:[1,2,3,4], y:[0,-1,-2,3]})) #Prints out the loss.. We are off by 23.66

#The correct answers for our linear model are -1 for W and 1 for b with 0 loss
#To get to this correct weights and bias and 0 loss, we need an optimizer to reduce the loss and update

optimizer = tf.train.GradientDescentOptimizer(0.01) #0.01 is what loss we are going for.
train = optimizer.minimize(loss)


sess.run(init) # initialize our variables

#reset values for incorrect defaults.
for i in range(1000):
	sess.run(train, {x:[1,2,3,4], y:[0,-1,-2,-3]})

print("Values for weights and bias after training: " + str(sess.run([W, b])))