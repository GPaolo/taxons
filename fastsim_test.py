# Created by Giuseppe Paolo 
# Date: 26/03/19

import gym, gym_fastsim
import numpy
import matplotlib.pyplot as plt

env = gym.make("FastsimSimpleNavigation-v0")
obs = env.reset()
print(obs)
display= True

plt.figure()

if(display):
	env.enable_display()

action = [1, 1]
for k in range(1000):
  o, r, eo, info = env.step(action)
  env.render()
  # plt.draw()
  print(info)
image = env.render(mode="rgb_array")
print(image.shape)
plt.imshow(image)
plt.show()


