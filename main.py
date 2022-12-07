import os
import json

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
import numpy as np
import random
from protopost import ProtoPost
from nd_to_json import json_to_nd

TARGET_LAYERS = os.getenv("TARGET_LAYERS", "[64, 64]")
TARGET_LAYERS = json.loads(TARGET_LAYERS)
PRED_LAYERS = os.getenv("PRED_LAYERS", "[64]")
PRED_LAYERS = json.loads(PRED_LAYERS)
OBS_SIZE = int(os.getenv("OBS_SIZE", 128))
FEATURE_SIZE = int(os.getenv("FEATURE_SIZE", 32))
PORT = int(os.getenv("PORT", 80))
TARGET_MODEL_PATH = os.getenv("TARGET_MODEL_PATH", "models/target")
PRED_MODEL_PATH = os.getenv("PRED_MODEL_PATH", "models/pred")
LEARNING_RATE = float(os.getenv("LEARNING_RATE", 0.00025))
ACTIVATION = os.getenv("ACTIVATION", "swish")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 64))
BUFFER_SIZE = int(os.getenv("BUFFER_SIZE", 1024))
AUTOSAVE_INTERVAL = int(os.getenv("AUTOSAVE_INTERVAL", 1000))
#TODO: reward and (better) observation normalization?

def create_model(layers):
    model = Sequential()
    model.add(Input([OBS_SIZE]))
    model.add(BatchNormalization())
    for size in layers:
        model.add(Dense(size, activation=ACTIVATION))
    model.add(Dense(FEATURE_SIZE))
    model.compile(loss="mse", optimizer=Adam(LEARNING_RATE))
    return model

try:
    print("Loading models", TARGET_MODEL_PATH, "and", PRED_MODEL_PATH)
    target_model = load_model(TARGET_MODEL_PATH)
    pred_model = load_model(PRED_MODEL_PATH)
    print("Models loaded")
except OSError as e:
    print(e)
    print('Model not found, or other ValueError occurred when loading model. Creating new model.')
    target_model = create_model(TARGET_LAYERS)
    pred_model = create_model(PRED_LAYERS)
    target_model.save(TARGET_MODEL_PATH)
    pred_model.save(PRED_MODEL_PATH)
    print("Models created")

print("Target model:")
target_model.summary()
print()
print("Prediction model:")
pred_model.summary()

buffer = []
step_counter = 0

def add_to_buffer(obs):
    global buffer
    buffer.append(obs)
    buffer = buffer[:BUFFER_SIZE]

def get_batch(size=32):
    return np.array(random.choices(buffer, k=size))

#calculate RND reward, train prediction network for 1 batch
def step(data):
    global step_counter
    obs = np.expand_dims(json_to_nd(data), 0)
    # print(obs)
    #feed obs to both networks
    y_true = target_model(obs)
    y_pred = pred_model(obs)
    #calculate reward
    reward = np.mean((y_true - y_pred) ** 2).item()
    #add obs to buffer
    add_to_buffer(obs[0])
    #get training batch
    batch_x = get_batch(BATCH_SIZE)
    batch_y = target_model(batch_x)
    #train prediction model
    loss = pred_model.train_on_batch(batch_x, batch_y)
    # print("training loss:", loss)
    #TODO: save network
    step_counter += 1
    if AUTOSAVE_INTERVAL > 0 and step_counter >= AUTOSAVE_INTERVAL:
        # print("saving...")
        step_counter = 0
        pred_model.save(PRED_MODEL_PATH)

    return reward

routes = {
    "": step
}

ProtoPost(routes).start(PORT)
