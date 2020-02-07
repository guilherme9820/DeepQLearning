class Config:

    INPUT_SHAPE = (84, 84, 4)

    BATCH_SIZE = 32

    """ Discount rate to be applied to Bellman's equation"""
    GAMMA = 0.99

    """ Number of frames that agent will train """
    TOTAL_FRAMES = 50000000
    EPISODES_TEST = 20

    """ Number of frames skipped per action """
    FRAME_SKIP = 4

    """ Optimizer learning rate"""
    LEARNING_RATE = 0.0001
    RHO = 0.99

    """ Target network update frequency (in number of frames) """
    TARGET_UPDATE = 100000

    """ Params for epsilon greedy policy"""
    EPSILON_MAX = 1.0
    EPSILON_MIN = 0.1
    DECAY_RATE = 0.002

    """ Actions to be performed by the agent """
    left = [1, 0, 0]
    right = [0, 1, 0]
    shoot = [0, 0, 1]
    ACTIONS = [left, right, shoot]

    """ Size of replay buffer """
    MEMORY_SIZE = 100000

    """ Initialize replay buffer with K random experiences """
    MEMORY_INIT_SIZE = 1000000

    """ Number of frames kept per state """
    STACK_SIZE = 4

    """ Save weights every X frames """
    SAVE_FREQ = 1000

    """ Maximum number of weight files that will be stored at WEIGHTS_DIR """
    MAX_SAVE = 10

    """ Path to doom scenario files """
    SCENARIOS_PATH = "./doom_scenarios/"

    """ Path to tensorboard logs """
    TRAIN_LOG_DIR = "./logs/gradient_tape/"

    """ Path to weight (.h5) files """
    WEIGHTS_DIR = "./weights/"
