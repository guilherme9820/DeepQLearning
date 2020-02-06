import numpy as np
import random
import tensorflow as tf


class EpsGreedyPolicy(tf.keras.layers.Layer):

    def __init__(self, min_eps, max_eps, decay_rate):
        super().__init__(name='eps_greedy')
        self._min_eps = min_eps
        self._max_eps = max_eps
        self._decay_rate = decay_rate

    def call(self, model, state_frame, possible_actions, decay_step):
        """ Select an action following the epsilon greedy policy """

        # Random value tha will decide if agent will explore or exploit
        tradeoff = np.random.rand()

        # Update epsilon
        epsilon = self._min_eps + (self._max_eps - self._min_eps) \
            * np.exp(-self._decay_rate * decay_step)

        if epsilon > tradeoff:
            # Take a random action to explore the environment
            action = random.choice(possible_actions)
        else:
            # Use known information from the agent to choose new action (Exploitation)
            action = model.evaluate_action(state_frame, possible_actions)

        return action
