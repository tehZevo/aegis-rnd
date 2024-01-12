# Aegis Random Network Distillation node
*Uses https://github.com/tehZevo/rnd for the RND implementation*

## Environment variables
- `PORT`: the port to listen on
- `INPUT_SIZE`: input (observation) vector size
- `OUTPUT_SIZE`: RND model output vector size
- `PREDICTOR_HIDDEN`: hidden layer sizes for the predictor network, separated by spaces e.g., `128 64` (defaults to `32`)
- `TARGET_HIDDEN`: as above, but for the target network, defaults to `32 32`
- `ACTIVATION`: hidden layer activation function for both models, defaults to "swish"
- `LR`: learning rate, defaults to 1e-3
- `BUFFER_SIZE`: training buffer size, defaults to 10000
- `BATCH_SIZE`: training batch size, defaults to 32
- `SAVE_PATH`: path to save/load models from, defaults to `models`
- `AUTOSAVE_STEPS`: save every <this many> steps, defaults to 1000
