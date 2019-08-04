import tensorflow as tf

class QNetwork:
    def __init__(self, learning_rate=0.01, state_size=18,
                 action_size=4, hidden_size=10,
                 name='QNetwork'):
        
        print('state size: {}, action size: {}'.format(state_size, action_size))
              
        # state inputs to the Q-network
        with tf.variable_scope(name):
            self.inputs_ = tf.placeholder(tf.float32, [None, state_size], name='inputs')
            
            # One hot encode the actions to later choose the Q-value for the action
            self.actions_ = tf.placeholder(tf.int32, [None, action_size], name='actions')
            
            # Target Q values for training
            self.targetQs_ = tf.placeholder(tf.float32, [None], name='target')
            print('target Qs: {}'.format(self.targetQs_))
            
            # ReLU hidden layers
            self.fc1 = tf.contrib.layers.fully_connected(self.inputs_, hidden_size)
            self.fc2 = tf.contrib.layers.fully_connected(self.fc1, hidden_size)
            
            # Linear output layer
            self.output = tf.contrib.layers.fully_connected(self.fc2, action_size, activation_fn=None)
            
            ### Train with loss (targetQ - Q)^2
            # output has length 4
            self.Q = tf.reduce_sum(self.output, axis=1)
            
            self.loss = tf.reduce_mean(tf.square(self.targetQs_ - self.Q))
            print('target Qs: {}'.format(self.targetQs_))
            print('Q: {}'.format(self.Q))
            print('loss: {}'.format(self.loss))
            
            self.opt = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)