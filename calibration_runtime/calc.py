import torch
import pickle
import os
import time
import subprocess

# --- Settings ---
layer_index = 0
input_path = "./test_output"
os.makedirs(input_path, exist_ok=True)

# --- Randomly initialize X and Y and save to xlayer_objs.pkl ---
torch.manual_seed(0)
d, n = 4096*2, 2048*256 # activation size, context length x sample size

X = torch.randn(d, n).to(torch.float32)
Y = torch.randn(d, n).to(torch.float32)

with open("xlayer_objs.pkl", "wb") as f:
    pickle.dump([X, Y], f)

del X, Y

start = time.time()
# --- Run the algorithm and time it ---

subprocess.run(["python3", "calculate_cca_gpu_parallel.py", str(layer_index), input_path], check=True)

elapsed = time.time() - start
print(f"\nTotal runtime: {elapsed:.2f} seconds")
