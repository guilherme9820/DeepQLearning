import numpy as np
import random
import glob
import os
import tensorflow as tf
from itertools import count


class QNet(tf.keras.Model):

    def __init__(self, name='qagent'):
        super().__init__(name=name)

    def configure(self, optimizer, loss, metric):
        self.metric = metric
        self.loss_fn = loss
        self.optimizer = optimizer

    def restore_weights(self, agent, file_name):
        agent.load_weights(file_name, by_name=True)

    @property
    def loss(self):
        loss = self.metric.result()
        self.metric.reset_states()
        return loss

    def call(self):
        return NotImplementedError()

    def evaluate_action(self):
        return NotImplementedError()


class DoubleDeepQNetwork(QNet):

    def __init__(self, gamma,
                 main_agent,
                 target_agent,
                 name='DeepQNetwork'):

        super().__init__(name=name)
        self._gamma = gamma

        self.main_agent = main_agent

        self.target_agent = target_agent

    def call(self, experiences):
        """ Updates main agent weights following "Deep Reinforcement Learning with Double Q-learning" (Hasselt et al., 2015) """

        state_batch, action_batch, reward_batch, \
            next_state_batch, done_batch = experiences

        with tf.GradientTape() as tape:

            predicted_next_q = self.target_agent(next_state_batch)

            target_q = reward_batch + (1 - done_batch) * \
                self._gamma * tf.reduce_max(predicted_next_q, axis=1)

            predicted_q = self.main_agent(state_batch)

            q = tf.reduce_sum(tf.multiply(predicted_q, action_batch), axis=1)

            loss = self.loss_fn(target_q, q)

        grads = tape.gradient(loss, self.main_agent.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.main_agent.trainable_weights))

        self.metric(loss)

    def evaluate_action(self, state_frame, possible_actions):
        """ Uses main agent to predict the new action 
            based on current frame sequence (current state)
        """
        prediction = self.main_agent(tf.expand_dims(state_frame, axis=0))
        choice = tf.argmax(prediction[0])
        return possible_actions[int(choice)]

    def synchronize_target(self):
        """ Updates target weights with main agent weights """
        self.target_agent.set_weights(self.main_agent.get_weights())
        print("Target synchronized")
