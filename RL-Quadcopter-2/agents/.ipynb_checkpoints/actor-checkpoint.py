{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/Users/chriskim/anaconda3/lib/python3.6/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.\n",
      "  from ._conv import register_converters as _register_converters\n",
      "Using TensorFlow backend.\n"
     ]
    }
   ],
   "source": [
    "from keras import layers, models, optimizers, initializers, regularizers\n",
    "from keras import backend as K\n",
    "import numpy as np\n",
    "import copy\n",
    "\n",
    "class Actor:\n",
    "    def __init__(self, state_space, action_space, action_min, action_max, hidden_units, learning_rate, q_lambda):\n",
    "        self.state_space = state_space\n",
    "        self.action_space = action_space\n",
    "        self.action_max = action_max\n",
    "        self.action_min = action_min\n",
    "        self.action_range = action_max - action_min\n",
    "        self.learning_rate = learning_rate\n",
    "        self.q_lambda = q_lambda\n",
    "        \n",
    "        var_wi = initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='uniform')\n",
    "        out_wi = initializers.RandomUniform(minval=-3e-3, maxval=3e-3)\n",
    "        \n",
    "        input_states = layers.Input(shape=(self.state_space,), name='input_states')\n",
    "        \n",
    "        layer = layers.Dense(units=400, kernel_regularizer=regularizers.l2(self.q_lambda))(input_states)\n",
    "        layer = layers.BatchNormalization()(layer)\n",
    "        layer = layers.Activation('relu')(layer)\n",
    "        \n",
    "        \n",
    "        layer = layers.Dense(units=300, kernel_regularizer=regularizer.l2(self.q_lambda))(layer)\n",
    "        layer = layers.BatchNormalization()(layer)\n",
    "        layer = layers.Activation('relu')(layer)\n",
    "        \n",
    "        norm_action = layers.Dense(self.action_space, kernel_initializer=out_wi, activation='sigmoid', name='norm_action')(layer)\n",
    "        \n",
    "        actions = layers.Lambda(lambda x: (x*self.action_range) + self.action_min, name='actions')(norm_action)\n",
    "        \n",
    "        self.model = model.Model(input=input_states, output=actions)\n",
    "        \n",
    "        action_gradients = layers.Input(shape=(self.action_space,))\n",
    "        loss = K.mean(-action_gradients * actions)\n",
    "        \n",
    "        adam_optimizer = optimizer.Adam(lr=self.learning_rate)\n",
    "        train_param = adam_optimizer.get_updates(params=self.model.traininable_weights, loss=loss)\n",
    "        \n",
    "        self.train_fn = K.function(inputs=[self.input, action_gradients, K.learning_phase()], outputs=[], updates=train_param)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
