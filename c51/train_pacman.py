
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import gym
import numpy as np
import pickle
import random

from gym import wrappers

# custom module imports
from config import C51DQNConfig
from c51 import C51DQN

def map_scores(dqfd_scores=None, ddqn_Scores=None, xlabel=None, ylabel=None):
    if dqfd_scores is not None:
        plt.plot(dqfd_scores, 'r')
    if ddqn_scores is not None:
        plt.plot(ddqn_scores, 'b')
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    plt.show()
