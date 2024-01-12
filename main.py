import os
import random
#suppress tf warnings for calling train_on_batch etc in quick succession
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
from protopost import ProtoPost
from nd_to_json import json_to_nd
from rnd import RND

PORT = int(os.getenv("PORT", "80"))
INPUT_SIZE = int(os.getenv("INPUT_SIZE", "32"))
OUTPUT_SIZE = int(os.getenv("OUTPUT_SIZE", "32"))
PREDICTOR_HIDDEN = os.getenv("PREDICTOR_HIDDEN", "32").split()
TARGET_HIDDEN = os.getenv("TARGET_HIDDEN", "32 32").split()
ACTIVATION = os.getenv("ACTIVATION", "swish")
LR = float(os.getenv("LR", "1e-3"))
BUFFER_SIZE = int(os.getenv("BUFFER_SIZE", "10000"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
SAVE_PATH = os.getenv("SAVE_PATH", "models")
AUTOSAVE_STEPS = int(os.getenv("AUTOSAVE_STEPS", 1000))

rnd = RND(
  input_size=INPUT_SIZE,
  output_size=OUTPUT_SIZE,
  predictor_hidden=PREDICTOR_HIDDEN,
  target_hidden=TARGET_HIDDEN,
  activation=ACTIVATION,
  lr=LR,
  buffer_size=BUFFER_SIZE,
  batch_size=BATCH_SIZE
)

#load if exists
if os.path.exists(os.path.join(SAVE_PATH, "predictor.keras")):
  print(f"Loading existing models from '{SAVE_PATH}'")
  rnd.load(SAVE_PATH)
#otherwise save
else:
  print(f"Saving new models to '{SAVE_PATH}'")
  rnd.save(SAVE_PATH)

print("RND Predictor summary")
rnd.predictor_model.summary()

print("RND Target summary")
rnd.target_model.summary()

steps = 0

def step(data):
  global steps

  x = json_to_nd(data)
  rnd_reward = rnd.step(x)

  #autosave
  if AUTOSAVE_STEPS >= 1:
    steps += 1
    if steps >= AUTOSAVE_STEPS:
      print(f"Saving to '{SAVE_PATH}'")
      rnd.save(SAVE_PATH)
      steps = 0

    return rnd_reward

routes = {
  "": step,
}

ProtoPost(routes).start(PORT)
