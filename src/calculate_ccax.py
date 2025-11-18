import os
import numpy as np
import torch
import pickle
import time
import sys
from multiprocessing import Pool

cache_file = sys.argv[1]
input_path = sys.argv[2]
skipped_layers = sys.argv[3]

print(skipped_layers)

if skipped_layers != "[]":
    skipped_layers = [int(i) for i in skipped_layers[1:-1].split(", ")]
else:
    skipped_layers = []

def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

torch.set_grad_enabled(False)

def process_layer(lyr):
    """
    Function to process a single layer and compute the required statistics.
    """
    with open(f"./llm_variables/xlayer_{lyr}_objs.pkl", "rb") as f:
        X, Y = pickle.load(f)
    
    start_time = time.time()
    print("1")
    X = X.type(torch.double)
    Y1 = Y.type(torch.double)
    Y = Y.type(torch.double) + X
    print("2")
    E_x = torch.mean(X, 1, keepdim=True)
    E_y = torch.mean(Y, 1, keepdim=True)
    E_y1 = torch.mean(Y1, 1, keepdim=True)
    print("3")
    Cyx = ((Y - E_y) @ (X - E_x).T) / (X.shape[1] - 1)
    Cyx1 = ((Y1 - E_y1) @ (X - E_x).T) / (X.shape[1] - 1)
    print("4")
    Cxx = torch.cov(X)
    Cyy = torch.cov(Y)
    print("5")
    eigenvalues_YY, eigenvectors_YY = torch.linalg.eigh(Cyy.type(torch.double))
    eigenvalues_XX, eigenvectors_XX = torch.linalg.eigh(Cxx.type(torch.double))
    print("6")
    C_YY_inv_sqrt = eigenvectors_YY @ torch.diag(eigenvalues_YY.pow(-0.5)) @ eigenvectors_YY.T
    C_XX_inv_sqrt = eigenvectors_XX @ torch.diag(eigenvalues_XX.pow(-0.5)) @ eigenvectors_XX.T
    corr = C_YY_inv_sqrt @ Cyx.type(torch.double) @ C_XX_inv_sqrt
    print("7")
    _, S, _ = torch.linalg.svd(corr, full_matrices=False)
    bound = torch.sum(1 - S**2)
    
    # Save statistics to a text file
    with open(input_path + "layerwise_statsx.txt", "a") as f:
        f.write(f"for layer {lyr} mse: \n")
        f.write("---------------------------------------------------- \n")
        f.write(f"sum bound: {bound * torch.trace(Cyy)}\n")
        f.write(f"sum sngs: {bound}\n")
        elapsed_time = time.time() - start_time
        f.write(f"Elapsed time: {elapsed_time}\n")
        f.write("---------------------------------------------------- \n")
    
    W = Cyx1 @ torch.linalg.inv(Cxx)
    b = E_y1 - W @ E_x

    # Save weights
    with open(input_path + f"xlayer_{lyr}_weights.pkl", 'wb') as f:
        pickle.dump([W, b], f)
    
    return lyr, bound

if __name__ == "__main__":
    similarities = torch.zeros(32,)
    print("Calculating CCA decompositions...")

    # Use a pool of workers to process layers in parallel
    lyr_list = []
    for i in range(32):
        if i not in skipped_layers:
            lyr_list.append(i)

    with Pool() as pool:
        results = pool.map(process_layer, lyr_list)

    # Collect results
    for lyr, bound in results:
        similarities[lyr] = 1/bound

    # Save similarities to a file
    with open(input_path + "similarity_scores", 'wb') as file:
        pickle.dump(similarities, file)

    print("similarities\n", similarities)

