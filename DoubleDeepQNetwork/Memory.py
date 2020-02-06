from collections import deque
import tensorflow as tf
import random


class Memory:

    def __init__(self, max_size, init_size=10000, max_steps=100):
        self.buffer = deque(maxlen=max_size)
        self._init_size = init_size
        self._max_steps = max_steps

    def add(self, experience):
        """ Adds new experience to replay memory """

        state = tf.cast(experience[0], tf.float32)
        action = tf.cast(experience[1], tf.float32)
        reward = tf.cast(experience[2], tf.float32)
        next_state = tf.cast(experience[3], tf.float32)
        done = tf.cast(experience[4], tf.float32)

        experience = (state, action, reward, next_state, done)

        self.buffer.append(experience)

    def sample(self, batch_size):
        """ Return samples of the memory buffer with size specified by batch_size"""

        batch = random.sample(self.buffer, batch_size)

        # Splits every feature into separated chunks, each feature have
        # shape (batch_size, feat_shape)
        state_batch = tf.stack([experience[0] for experience in batch])
        action_batch = tf.stack([experience[1] for experience in batch])
        reward_batch = tf.stack([experience[2] for experience in batch])
        next_state_batch = tf.stack([experience[3] for experience in batch])
        done_batch = tf.stack([experience[4] for experience in batch])

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def initialize(self, environment):
        """ Initialize memory with random experiences """

        environment.init_game()

        while len(self) < self._init_size:

            environment.reset_frame_sequence()

            # Generate a maximum of 100 frames per episode
            for _ in range(self._max_steps):

                experience = environment.generate_experience()

                self.add(experience)

                done = experience[-1]

                if done or (len(self) == self._init_size):
                    break

        environment.close_game()

    def __len__(self):
        """ Returns the length of replay memory when casting len() 
            function over a instance of this class
        """
        return len(self.buffer)
