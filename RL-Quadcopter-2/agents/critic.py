from keras import layers, models, optimizers, initializers, regularizers
from keras import backend as K
import numpy as np
import copy

class Critic:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
        self.build_model()
        
    def build_model(self):
        # States Input layer
        states = layers.Input(shape=(self.state_size,), name='states')
        
        # State hidden layer
        layer_states = layers.Dense(units=400, kernel_regularizer=regularizers.l2(1e-6))(states)
        layer_states = layers.BatchNormalization()(layer_states)
        layer_states = layers.Activation('relu')(layer_states)
        
        layer_states = layers.Dense(units=300, activation='relu', kernel_regularizer=regularizers.l2(1e-6))(layer_states)
        
        # Actions Input layer
        actions = layers.Input(shape=(self.action_size,), name='actions')
        
        # Action hidden layer
        layer_actions = layers.Dense(units=300, activation='relu', kernel_regularizer=regularizers.l2(1e-6))(actions)
        
        # Advantage network
        layer = layers.Add()([layer_states, layer_actions])
        layer = layers.Activation('relu')(layer)
        
        # Output layer to produce Q values
        kernel_initializer = initializers.RandomUniform(minval=-0.003, maxval=0.003)
        q_values = layers.Dense(units=1, kernel_initializer=kernel_initializer, name='q_values')(layer)
        
        # Create Keras model
        self.model = models.Model(inputs=[states, actions], outputs=q_values)
        
        optimizer = optimizers.Adam(lr=.001)
        self.model.compile(loss='mean_squared_error', optimizer=optimizer)
        
        # Define function to get action gradients
        action_gradients = K.gradients(loss=q_values, variables=actions)
        
        self.get_action_gradients = K.function(inputs=[*self.model.input, K.learning_phase()], outputs=action_gradients)
        
        print('critic model')
        self.model.summary()
      
        