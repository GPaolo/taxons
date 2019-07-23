# Created by Giuseppe Paolo 
# Date: 26/03/19

import gym, gym_fastsim
import numpy

env = gym.make("FastsimSimpleNavigation-v0")
obs = env.reset()
print(obs)
display= True

if(display):
	env.enable_display()

action = [-0.01, 0.01]
for k in range(10000):
  o, r, eo, info = env.step(action)
  env.render()
  print(info)


