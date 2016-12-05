import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.ops import rnn, rnn_cell
import _infimnist as infimnist
import numpy as np
mnist = infimnist.InfimnistGenerator()
#mnist = input_data.read_data_sets("/mnt/d/Py_TF/data/infimnist", one_hot = True)

train_num_examples = 1000000
#indexes = np.arange(10000,(train_num_examples+9999),dtype=np.int64)
testindexes = np.arange(0,9999,dtype=np.int64)

hm_epochs = 11
n_classes = 10
batch_size = 4096
chunk_size = 28
n_chunks = 28
rnn_size = 2048

#tr_digits, tr_labels = mnist.gen(indexes)
te_digits, te_labels = mnist.gen(testindexes)

x = tf.placeholder('float', [None, n_chunks,chunk_size])
y = tf.placeholder('float')

def recurrent_neural_network(x):
    layer = {'weights':tf.Variable(tf.random_normal([rnn_size,n_classes])),
             'biases':tf.Variable(tf.random_normal([n_classes]))}

    x = tf.transpose(x, [1,0,2])
    x = tf.reshape(x, [-1, chunk_size])
    x = tf.split(0, n_chunks, x)

    lstm_cell = rnn_cell.BasicLSTMCell(rnn_size)#,state_is_tuple=True)
    stacked_lstm = rnn_cell.MultiRNNCell([lstm_cell] * 2)
    outputs, states = rnn.rnn(stacked_lstm, x, dtype=tf.float32)

    output = tf.matmul(outputs[-1],layer['weights']) + layer['biases']

    return output


def preprocess_y(y):
    output=np.array([[]])
    for i in y:
        if(i == 0):
            output = np.append(output,[1,0,0,0,0,0,0,0,0,0])
        elif(i==1):
            output = np.append(output,[0,1,0,0,0,0,0,0,0,0])
        elif(i==2):
            output = np.append(output,[0,0,1,0,0,0,0,0,0,0])
        elif(i==3):
            output = np.append(output,[0,0,0,1,0,0,0,0,0,0])
        elif(i==4):
            output = np.append(output,[0,0,0,0,1,0,0,0,0,0])
        elif(i==5):
            output = np.append(output,[0,0,0,0,0,1,0,0,0,0])
        elif(i==6):
            output = np.append(output,[0,0,0,0,0,0,1,0,0,0])
        elif(i==7):
            output = np.append(output,[0,0,0,0,0,0,0,1,0,0])
        elif(i==8):
            output = np.append(output,[0,0,0,0,0,0,0,0,1,0])
        elif(i==9):
            output = np.append(output,[0,0,0,0,0,0,0,0,0,1])
    return output

def train_neural_network(x):
    prediction = recurrent_neural_network(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction,y) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        #saver.restore(sess,'model5.ckpt')

        for epoch in [0]:
            epoch_loss = 0
            batch_size = pow(2,(epoch+1))
            batch_cs = 10000
            for batch_c in range(int(train_num_examples/batch_size)):
                epoch_x,epoch_y = mnist.gen(np.arange(batch_cs,(batch_cs+batch_size),dtype=np.int64))
                batch_cs = batch_cs + batch_size
                epoch_y = preprocess_y(epoch_y)
                epoch_y = epoch_y.reshape((batch_size,n_classes))
                epoch_x = epoch_x.reshape((batch_size,n_chunks,chunk_size))

                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                print('Minibatch:',batch_c,'BatchSize:',batch_size,'Loss:',c)
                epoch_loss += c
            epoch_loss = "%.4f" % (1.0-(epoch_loss/(train_num_examples/batch_size)))           
            print('Epoch', (epoch+1), 'completed out of',hm_epochs,'Accuracy:',epoch_loss)
            modelFile = "model%d.ckpt" % epoch
            logFile = "log%d.txt" % epoch
            with open(logFile,'w') as f:
            	f.write(epoch_loss)
            saver.save(sess,modelFile)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:',accuracy.eval({x:te_digits.reshape((-1, n_chunks, chunk_size)), y:te_labels}))

train_neural_network(x)
