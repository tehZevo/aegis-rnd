import numpy as np
from protopost import protopost_client as ppcl
from nd_to_json import nd_to_json

HOST = "http://127.0.0.1:80"
OBS_SIZE = 128

RND = lambda obs: ppcl(HOST, nd_to_json(obs))

obs = np.random.normal(size=[OBS_SIZE])

for i in range(100):
    reward = RND(obs)
    print(f"Step {i} reward: {reward}")
