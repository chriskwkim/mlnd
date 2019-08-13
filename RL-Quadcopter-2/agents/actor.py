from keras import layers, models, optimizers, initializers, regularizers
from keras import backend as K
import numpy as np
import copy

class Actor:
    def __init__(self, state_space, action_space, action_min, action_max, hidden_units, learning_rate, q_lambda):
        self.state_space = state_space
        self.action_space = action_space
        self.action_max = action_max
        self.action_min = action_min
        self.action_range = action_max - action_min
        self.learning_rate = learning_rate
        self.q_lambda = q_lambda
        
        var_wi = initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='uniform')
        out_wi = initializers.RandomUniform(minval=-3e-3, maxval=3e-3)
        
        input_states = layers.Input(shape=(self.state_space,), name='input_states')
        
        layer = layers.Dense(units=400, kernel_regularizer=regularizers.l2(self.q_lambda))(input_states)
        layer = layers.BatchNormalization()(layer)
        layer = layers.Activation('relu')(layer)
        
        
        layer = layers.Dense(units=300, kernel_regularizer=regularizers.l2(self.q_lambda))(layer)
        layer = layers.BatchNormalization()(layer)
        layer = layers.Activation('relu')(layer)
        
        norm_action = layers.Dense(self.action_space, kernel_initializer=out_wi, activation='sigmoid', name='norm_action')(layer)
        
        actions = layers.Lambda(lambda x: (x*self.action_range) + self.action_min, name='actions')(norm_action)
        
        self.model = models.Model(input=input_states, output=actions)
        
        action_gradients = layers.Input(shape=(self.action_space,))
        loss = K.mean(-action_gradients * actions)
        
        adam_optimizer = optimizers.Adam(lr=self.learning_rate)
        train_param = adam_optimizer.get_updates(params=self.model.trainable_weights, loss=loss)
        
        self.train_fn = K.function(inputs=[self.model.input, action_gradients, K.learning_phase()], outputs=[], updates=train_param)