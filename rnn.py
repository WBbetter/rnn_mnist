import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.ops import rnn, rnn_cell
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

hm_epochs = 5
n_classes = 10
batch_size = 4096
chunk_size = 28
n_chunks = 28
rnn_size = 256


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



def train_neural_network(x):
    prediction = recurrent_neural_network(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction,y) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        saver.restore(sess,'model5.ckpt')

        for epoch in [0,1,2,3,4,5,6,7]:
            epoch_loss = 0
            batch_size = pow(2,(epoch+1))
            for batch_c in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                print epoch_x.shape
                epoch_x = epoch_x.reshape((batch_size,n_chunks,chunk_size))
                print epoch_x.shape

                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                print('Minibatch:',batch_c,'BatchSize:',batch_size,'Loss:',c)
                epoch_loss += c
            epoch_loss = "%.4f" % (1.0-(epoch_loss/(mnist.train.num_examples/batch_size)))           
            print('Epoch', (epoch+1), 'completed out of',hm_epochs,'Accuracy:',epoch_loss)
            modelFile = "model%d.ckpt" % epoch
            logFile = "log%d.txt" % epoch
            with open(logFile,'w') as f:
            	f.write(epoch_loss)
            saver.save(sess,modelFile)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:',accuracy.eval({x:mnist.test.images.reshape((-1, n_chunks, chunk_size)), y:mnist.test.labels}))

train_neural_network(x)
