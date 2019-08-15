from keras import layers, models, optimizers, initializers, regularizers
from keras import backend as K
import numpy as np
import copy

class Actor:
    def __init__(self, state_size, action_size, action_low, action_high):
        self.state_size = state_size
        self.action_size = action_size
        self.action_high = action_high
        self.action_low = action_low
        self.action_range = action_high - action_low
        
        self.build_model()
        
    def build_model(self):
        """Build Actor network"""
        # Input layer
        states = layers.Input(shape=(self.state_size,), name='states')
        
        # Hidden layers
        layer = layers.Dense(units=400, kernel_regularizer=regularizers.l2(1e-6))(states)
        layer = layers.BatchNormalization()(layer)
        layer = layers.Activation('relu')(layer)
        
        layer = layers.Dense(units=300, kernel_regularizer=regularizers.l2(1e-6))(layer)
        layer = layers.BatchNormalization()(layer)
        layer = layers.Activation('relu')(layer)
        
        # Output layer
        kernel_initializer = layers.initializers.RandomUniform(minval=-0.003, maxval=0.003)
        norm_action = layers.Dense(self.action_size, kernel_initializer=kernel_initializer, activation='sigmoid', name='norm_action')(layer)
        
        # Scale action values to proper range
        actions = layers.Lambda(lambda x: (x*self.action_range) + self.action_low, name='actions')(norm_action)
        
        # Create Keras model
        self.model = models.Model(inputs=states, outputs=actions)
        
        # Loss function
        action_gradients = layers.Input(shape=(self.action_size,))
        loss = K.mean(-action_gradients * actions)
        
        # Optimizer and training function
        optimizer = optimizers.Adam(lr=.0001)
        updates = optimizer.get_updates(params=self.model.trainable_weights, loss=loss)
        
        self.train_fn = K.function(inputs=[self.model.input, action_gradients, K.learning_phase()], outputs=[], updates=updates)
        
        print('actor model')
        self.model.summary()
        