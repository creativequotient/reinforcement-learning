# reinforcement-learning in *pytorch*
This repository is code written by myself as a way to gain familiarity and mastery over basic reinforcement learning algorithms. 
Much of the algorithms here are adapted from OpenAI's spinningup or baselines. 
This library will be implemented using **pytorch**.

# Solved environments
## LunarLanderContinuous-v2
LunarLanderContinuous-v2 solved was solved in ~260 episodes using DDPG. 

To reproduce, execute the following 
`python train.py --env LunarLanderContinuous-v2 --tau 0.001`.

To view pre-trained agents, execute `python test.py experiments/LunarLanderContinuous-v2/luna_v2 --render` or `luna_v1`.


`luna_v2` was trained in 270 episodes and has a 100-episode average of 250 whereas `luna_v1` has a 100-episode average of only 220.

## Pendulum-v0
Pendulum-v2 has no `solved` threshold, however, it clearly achieves its goal of keeping the pendulum upright when reviewed visually. 

To reproduce, execute
`python train.py --env Pendulum-v0`.

To view pre-trained agents, execute `python test.py experiments/Pendulum-v0/DDPG_1 --render` or `DDPG_2`. Both solve the environment, though `DDPG_1` was trained 'incorrectly' due to unscaled actions (ie action range was [-1,1] instead of [-2,2]).

## MountainCarContinuous-v0
MountainCarContinuous-v0 solved at ~1000 episodes, possibly sooner depending on seed due to need for "successful" exploration of the left slope. 

To reproduce, execute `python train.py --env MountainCarContinuous-v0`.

To view pre-trained agents, execute `python test.py experiments/MountainCarContinuous-v0/DDPG_1 --render`.

## BipedalWalker-v3
BipedalWalker-v3 unsolved but gets very close to solving. Consistently hits the upper 290 points but rarely goes to 300.

To reproduce, execute `python train.py --env BipedalWalker-v3 --q_lr 0.002`.

To view pre-trained agent, execute `python test.py -r experiments/BipedalWalker-v3/DDPG_1 --render`.

# Dependencies
`python 3.6.10`\
`gym 0.16.0`\
`numpy 1.18.1`\
`torch 1.3.1`

CUDA Specific dependencies:

`cudnn 7.6.5`\
`cudatoolkit 10.1.243`

# TODO
## High priority
- [ ] Implement PPO and TRPO policies
- [ ] Change `ReplayBuffer` to create `replay_buffer_size` length of zero arrays as a placeholder so as to avoid out of memory errors as the `ReplayBuffer` fills up

## Medium priority
- [ ] Add ability to resume training from pre-trained models

## Low priority