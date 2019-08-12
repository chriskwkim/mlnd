from keras import layers, models, optimizers, initializers, regularizers
from keras import backend as K
import numpy as np
import copy

class Critic:
    def __init__(self, state_space, action_space, hidden_units, learning_rate, q_lambda):
        self.state_space = state_space
        self.action_space = action_space
        self.hidden_units = hidden_units
        self.learning_rate = learning_rate
        self.q_lambda = q_lambda
        
        var_wi = initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='uniform')
        out_wi = initializers.RandomUniform(minval=-3e-3, maxval=3e-3)
        
        # States network
        input_states = layers.Input(shape=(self.state_space,), name='input_states')
        
        layer_states = layers.Dense(units=400, kernel_regularizer=regularizers.l2(self.q_lambda))(input_states)
        layer_states = layers.BatchNormalization()(layer_states)
        layer_states = layers.Activation('relu')(layer_states)
        
        layer_states = layers.Dense(units=300, activation='relu', kernel_regularizer=regularizers.l2(self.q_lambda))(layer_states)
        
        # Action network
        input_actions = layers.Input(shape=(self.action_space,), name='input_actions')
        
        layer_actions = layers.Dense(units=300, activation='relu', kernel_regularizer=regularizers.l2(self.q_lambda))(input_actions)
        
        
        # Advantage network
        layer = layers.Add()([layer_states, layer_actions])
        layer = layers.Activation('relu')(layer)
        
        q_values = layers.Dense(units=1, kernel_initializer=out_wi, name='q_values')(layer)
        
        
        self.model = models.Model(inputs=[input_states, input_actions], outputs=q_values)
        
        adam_optimizer = optimizers.Adam(lr=self.learning_rate)
        self.model.compile(loss='mean_squared_error', optimizer=adam_optimizer)
        
        # Define function to get action gradients
        action_gradients = K.gradients(loss=q_values, variables=input_action)
        self.get_action_gradients = K.function(inputs=[*self.model.input, K.learning_phase()], outputs=action_gradients)
        