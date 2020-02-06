import numpy as np
from vizdoom import *
import time
import os
import random
import copy
from collections import deque


class Environment:

    def __init__(self, possible_actions, process_alg, stack_size=4, frame_skip=4):
        self._process_alg = process_alg
        self._stack_size = stack_size
        self._frame_skip = frame_skip
        self._actions = possible_actions
        self._frame_sequence = deque(maxlen=stack_size)

    def generate_experience(self):
        """ Select a random action, apply it to the game and return 
            the resulting experience
        """

        curr_sequence = copy.deepcopy(self._frame_sequence)
        curr_sequence = self._process_alg(curr_sequence)

        action = random.choice(self._actions)

        next_state_frame, reward, done = self.frame_skip(action)

        experience = (curr_sequence, action, reward, next_state_frame, done)

        return experience

    def frame_skip(self, action):
        """ Perform same action over K subsequent frames"""

        acc_reward = 0

        frame_shape = self._frame_sequence[0].shape
        skipped_frames = self._frame_skip

        while 0 < skipped_frames:

            reward, done = self.perform_action(action)

            # Accumulated reward
            acc_reward += reward

            if done:
                break

            skipped_frames -= 1

        if skipped_frames != 0:
            # If episode ends within the frame skipping interval then we must fill the
            # frame sequence with black screens corresponding to the number of "not" skipped
            # frames, i.e., considering that our frame skipping window is 4, if episode
            # ends after the first action then we must fill the frame sequence with 4
            # black screens, if ends after the second action then fill with 3 black screens
            # and so on.
            black_screens = [np.zeros(frame_shape, dtype=np.float32)] * skipped_frames
            self._frame_sequence.extend(black_screens)
            next_state_frame = self._process_alg(self._frame_sequence)
        else:
            # If episode does not end within the frame skipping interval then just get the
            # next frame
            next_state_frame = self.get_new_frames()

        acc_reward = self.reward_normalization(acc_reward)

        return next_state_frame, acc_reward, done

    def test_environment(self, episodes):
        """ Run environment over few episodes to check if it is
            running properly
        """

        self.configure()

        self.init_game()

        for episode in range(episodes):
            done = False

            self.reset_frame_sequence()

            while not done:

                _ = self.get_new_frames()

                action = random.choice(self._actions)

                reward, done = self.perform_action(action)

                time.sleep(0.02)  # 50 fps

            print("Episode reward: ", self.get_episode_reward())
            time.sleep(2)

        self.close_game()

    # Normalize reward to range [-1, 1]
    def reward_normalization(self, reward):
        return (2 * (reward - self.min_reward) / (self.max_reward - self.min_reward)) - 1

    @property
    def actions(self):
        return self._actions

    def configure(self):
        return NotImplementedError()

    def reset_frame_sequence(self):
        return NotImplementedError()

    def get_new_frames(self):
        return NotImplementedError()

    def perform_action(self):
        return NotImplementedError()

    def get_episode_reward(self):
        return NotImplementedError()

    def init_game(self):
        return NotImplementedError()

    def close_game(self):
        return NotImplementedError()


class DoomEnvironment(Environment):

    def __init__(self, process_alg, possible_actions, scenario_path=None, stack_size=4, frame_skip=4):
        super().__init__(possible_actions=possible_actions,
                         process_alg=process_alg,
                         stack_size=stack_size,
                         frame_skip=frame_skip)

        self._scenario_path = scenario_path
        self._game = DoomGame()

    def configure(self, min_reward=None, max_reward=None, scenario='basic', render_screen=False):
        cfg_file = os.path.join(self._scenario_path, scenario + '.cfg')
        wad_file = os.path.join(self._scenario_path, scenario + '.wad')
        self.min_reward = min_reward or 0
        self.max_reward = max_reward or 1

        self._game.load_config(cfg_file)
        self._game.set_doom_scenario_path(wad_file)
        self._game.set_window_visible(render_screen)

    def reset_frame_sequence(self):
        """ Start a new episode of the game and 'clean' the frame buffer """

        self._game.new_episode()

        # Get current frame from game
        curr_frame = self._game.get_state().screen_buffer

        # Fill every position of the frame buffer with the same image
        self._frame_sequence.extend([curr_frame] * self._stack_size)

    def get_new_frames(self):
        """ Get a new stack of frames"""

        state_frame = self._game.get_state().screen_buffer

        self._frame_sequence.append(state_frame)

        # Apply the image processing algorithm over the frame buffer
        return self._process_alg(self._frame_sequence)

    def perform_action(self, action):
        reward = self._game.make_action(action)
        done = self._game.is_episode_finished()

        return reward, done

    def get_episode_reward(self):
        return self._game.get_total_reward()

    def init_game(self):
        self._game.init()

    def close_game(self):
        self._game.close()

    def test_frame_sequence(self):

        import matplotlib.pyplot as plt
        import random

        done = False

        self.init_game()

        self.reset_frame_sequence()

        for i in range(10):

            action = random.choice(self._actions)

            next_state_frame, reward, done = self.frame_skip(action)

            plt.figure(i)
            plt.subplot(141)
            plt.imshow(next_state_frame[:, :, 0], cmap='gray')
            plt.subplot(142)
            plt.imshow(next_state_frame[:, :, 1], cmap='gray')
            plt.subplot(143)
            plt.imshow(next_state_frame[:, :, 2], cmap='gray')
            plt.subplot(144)
            plt.imshow(next_state_frame[:, :, 3], cmap='gray')

        self.close_game()

        plt.show()
        input()
        plt.close('all')
