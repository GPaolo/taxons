# Created by Giuseppe Paolo 
# Date: 26/03/19

import multiprocessing as mp
import random, string
import time


random.seed(7)

alpf = ['a', 'b','c','d','e','f','g','h','i','l','m','n','o','p','q','r','s','t','u','v','z']

def rand_string(tt):
  rand_str = ''.join(random.choice(string.ascii_lowercase + string.ascii_uppercase + string.digits) for i in range(3))
  rand_str = "{}_{}_{}".format(tt[0], tt[1], rand_str)
  return rand_str


pool = mp.Pool(processes=4)

results = pool.map(rand_string, enumerate(alpf))
print(results)
