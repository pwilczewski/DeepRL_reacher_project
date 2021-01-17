
# Solution to Project 2: Continuous Control

### Introduction

This repository contains my solution for the second project in Udacity's Deep Reinforcement Learning Nanodegree. For this project I trained an agent to solve the Reacher environment by controlling a double-joined arm. In this environment a reward of +0.1 is provided for each step that the agent's hand is in the target location. The goal of my agent is to maintain its position at the target location for as many time steps as possible.

The observation space includes 33 variables corresponding to the position, velocity and angular velocities of the arm. Each action is a vector [-1,1] corresponding to the torque applied to the agent's two joints. The environment is considered solved once the agent achieves an average score of +30 over 100 consecutive episodes. There are two versions of the environment available, the first for a single agent and the second for twenty identical agents. This solution is for the single agent version.

### Getting Started

My solution was coded using Python version 3.6.12, PyTorch version 0.4.0 and OpenAI gym version 0.17.3.

1. The requirements for running my solution are available in the Udacity [DRLND GitHub repository](https://github.com/udacity/deep-reinforcement-learning#dependencies). These instructions can be found in `README.md` at the root of the repository.

2. Additionally you will need to download the One Agent environment. My solution was created using Windows (64-bit).

    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

3. After downloading the file you may copy this repository, place the Reacher environment file in the project directory and unzip it. 

### Instructions

After completing the initial setup, the Jupyter Notebook `Continuous_Control_Solution.ipynb` contains my solution. The notebook references two supporting py files. The `udacity_agent.py` file contains the definition of the agent and the `udacity_model.py` file contains the structure of the deep neural network model. Finally the estimated model parameters that solved the environment are located in the `actor_local.pth` and `critic_local.pth` files. The `report.md` file contains a descrption of the algorithm used to estimate these parameters.
