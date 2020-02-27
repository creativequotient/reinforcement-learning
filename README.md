# CarRacing

Attempt to solve OpenAI Gym's CarRacing-v0 environment using DDPG (Deep Deterministic Policy Gradients) algorithm.

Each action is repeated for the next 8 frames in the environment and 4 frames are stacked together and passed to the neural networks as inputs to enable perception of velocity.