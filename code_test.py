# Created by Giuseppe Paolo 
# Date: 26/03/19

import gym, gym_fastsim, pybulletgym
import numpy
import matplotlib.pyplot as plt

env = gym.make("AntMuJoCoEnv-v0")
env.render()
obs = env.reset()
print(obs)

plt.figure()


for k in range(2000):
  o, r, eo, info = env.step(env.action_space.sample())
  env.render()
  # plt.draw()
  print(env.robot.body_xyz)
image = env.render(mode="rgb_array", top_bottom=True)
print(image.dtype)
plt.imshow(image)
plt.show()


