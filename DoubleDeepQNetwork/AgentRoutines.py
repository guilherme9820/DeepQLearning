from itertools import count
import tensorflow as tf
import numpy as np


def train_agent(model,
                environment,
                total_frames,
                batch_size,
                policy,
                target_update,
                memory,
                summary,
                weight_saver):
    """ Train agents over a specified number of frames """

    environment.init_game()


    counter = count()
    
    # Set frame counter to 0
    frame_count = next(counter)

    # Train agent by a total of K frames specified by 'total_frames' parameter
    while frame_count < total_frames:

        environment.reset_frame_sequence()
        done = False

        state_frames = environment.get_new_frames()

        while not done:

            # Applies policy to choose an action
            action = policy(model, state_frames, environment.actions, frame_count)

            next_state_frames, reward, done = environment.frame_skip(action)

            # Increments number of frames
            frame_count = next(counter)

            experience = (state_frames, action, reward, next_state_frames, done)
            memory.add(experience)

            state_frames = next_state_frames

            model(memory.sample(batch_size))

            # Saves weights of the agent
            weight_saver(model.main_agent, frame_count)

        # Updates target weights after K frames have been read
        if frame_count % target_update == 0:
            model.synchronize_target()

        reward = environment.get_episode_reward()

        loss = model.loss

        print(f"# of frames: {frame_count}; Total reward: {reward}; Loss: {loss}; Buffer size: {len(memory)}")

        # Saves loss and reward into tensorflow summary
        with summary.as_default():
            tf.summary.scalar('loss', loss, step=frame_count)
            tf.summary.scalar('reward', reward, step=frame_count)

    environment.close_game()


def test_agent(model, environment, episodes):
    """ Test the trained agent over few episodes """

    environment.init_game()

    for episode in range(episodes):

        done = False

        environment.reset_frame_sequence()

        state_frames = environment.get_new_frames()

        while not done:

            action = model.evaluate_action(state_frames, environment.actions)

            _, done = environment.perform_action(action)

            if not done:
                state_frames = environment.get_new_frames()

        print(f"Episode {episode}:\tTotal reward: {environment.get_episode_reward()}")

    environment.close_game()
