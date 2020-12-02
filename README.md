# RL-project-repo

This repository contains implementations of DQN and C51 on the classic
Atari game called Pacman. We used the environment provided by OpenAI Gym.
The goal of this project was to reproduce results first introduced in
[this](https://arxiv.org/pdf/1707.06887.pdf) paper and become familiarized
with the simple single network Deep RL algorithms. We borrowed and adapted
the code from repositories mentioned in the References section.

Team:

Ronak Mehta (DQN)

Manmeet Singh (Categorical DQN)

Folder Structure:

assets: contains static artifacts, primarily plots of results from logs

colabs: notebooks with execution code and tensorboard visualzations

logs: a detailed list of metrics logged per frame to calculate performance

Models: saved models in h5 and pickle formats

training_videos: video captures of episodic training saved every n videos

Presentation.pdf

Code References: 
*https://github.com/MichaelSpencerA/atari-breakout-dqn
*https://github.com/flyyufelix/C51-DDQN-Keras
*https://github.com/Jash-R/ReinforcmentLearning-For-MsPacman (original starter code which fails to converge and contains bugs)

