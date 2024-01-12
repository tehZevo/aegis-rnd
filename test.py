import os
import random

import numpy as np
from protopost import protopost_client as ppcl
from nd_to_json import nd_to_json

INPUT_SIZE = 32
HOST = "http://127.0.0.1:8080"

RND = lambda obs: ppcl(HOST, nd_to_json(obs))

#by using a seed, repeated runs should have low rnd reward
np.random.seed(777)
samples = np.random.normal(0, 1, size=[10, INPUT_SIZE])

#rnd reward should decrease
for step in range(100):
  sample = random.choice(samples)
  rnd_reward = RND(sample)
  print(f"Step {step + 1} reward: {rnd_reward}")
