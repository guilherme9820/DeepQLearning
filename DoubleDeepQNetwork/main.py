from DoomEnvironment import DoomEnvironment
from Utils import *
from Memory import Memory
from QNetworks import DoubleDeepQNetwork
from AgentRoutines import *
from AgentArchitectures import *
from Config import Config
from Policy import EpsGreedyPolicy
from tensorflow.keras.optimizers import RMSprop
import warnings
import os
import tensorflow as tf
import datetime

warnings.filterwarnings('ignore')

config = Config()

if not os.path.exists(config.TRAIN_LOG_DIR):
    os.makedirs(config.TRAIN_LOG_DIR)

if not os.path.exists(config.WEIGHTS_DIR):
    os.mkdir(config.WEIGHTS_DIR)

TRAINING = False

doom_env = DoomEnvironment(preprocess_frames,
                           config.ACTIONS,
                           config.SCENARIOS_PATH,
                           config.STACK_SIZE,
                           config.FRAME_SKIP)

# Policy network
main_agent = DeepQNetArch(len(config.ACTIONS), input_shape=config.INPUT_SHAPE, name='policy_net')
main_agent.initialize_weights()

if TRAINING:

    # optimizer = Adam(config.LEARNING_RATE)
    optimizer = RMSprop(learning_rate=config.LEARNING_RATE, rho=config.RHO)
    loss = tf.keras.losses.Huber()
    metric = tf.keras.metrics.Mean()

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = config.TRAIN_LOG_DIR + current_time + '/train'
    summary_writer = tf.summary.create_file_writer(train_log_dir)

    weight_saver = WeightSaver(save_freq=config.SAVE_FREQ,
                               weights_dir=config.WEIGHTS_DIR,
                               max_save=config.MAX_SAVE)

    # Defines the enviroment scenario and whether the window will be rendered or not
    doom_env.configure(min_reward=-410, max_reward=95, scenario='basic', render_screen=False)

    # Policy for action selection
    eps_policy = EpsGreedyPolicy(config.EPSILON_MIN, config.EPSILON_MAX, config.DECAY_RATE)

    # Target network
    target_agent = DeepQNetArch(len(config.ACTIONS), input_shape=config.INPUT_SHAPE, name='target_net')
    target_agent.initialize_weights()

    memory = Memory(config.MEMORY_SIZE, config.MEMORY_INIT_SIZE)
    memory.initialize(doom_env)

    ddqn = DoubleDeepQNetwork(config.GAMMA, main_agent, target_agent)
    ddqn.configure(optimizer=optimizer, loss=loss, metric=metric)

    restore_last_saved(ddqn, ddqn.main_agent, config.WEIGHTS_DIR)
    restore_last_saved(ddqn, ddqn.target_agent, config.WEIGHTS_DIR)

    train_agent(ddqn,
                doom_env,
                config.TOTAL_FRAMES,
                config.BATCH_SIZE,
                eps_policy,
                config.TARGET_UPDATE,
                memory,
                summary_writer,
                weight_saver)

else:

    # Defines the enviroment scenario and whether the window will be rendered or not
    doom_env.configure(min_reward=None, max_reward=None, scenario='basic', render_screen=True)

    ddqn = DoubleDeepQNetwork(None, main_agent, None)

    restore_last_saved(ddqn, ddqn.main_agent, config.WEIGHTS_DIR)

    # doom_env.test_frame_sequence()

    test_agent(ddqn, doom_env, config.EPISODES_TEST)
