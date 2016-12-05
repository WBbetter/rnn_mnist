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



def test_neural_network(x):
    prediction = recurrent_neural_network(x)
    saver = tf.train.Saver()    
    with tf.Session() as sess:        
        sess.run(tf.initialize_all_variables())
        modelFile = "model.ckpt"
        saver.restore(sess,modelFile)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:',accuracy.eval({x:mnist.test.images.reshape((-1, n_chunks, chunk_size)), y:mnist.test.labels}),'ModelNo.',i)

test_neural_network(x)