import os
import torch
import pickle
import time
import sys
import torch.nn as nn
#from awq.modules.linear import WQLinear_GEMM  # adjust this if needed


layer_index = int(sys.argv[1])
input_path = sys.argv[2]

def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

torch.set_grad_enabled(False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""def activation_aware_quantize_linear(linear, sample_input, sample_output, w_bit=4, group_size=128):

    device = linear.weight.device
    in_features = linear.in_features
    out_features = linear.out_features

    # Transpose to group over input features
    W = linear.weight.data.t().contiguous()  # shape: (in_features, out_features)
    G = in_features // group_size
    W_q = torch.zeros_like(W)
    scales = torch.zeros((G, out_features), dtype=torch.float32, device=device)
    zeros = torch.zeros((G, out_features), dtype=torch.int32, device=device)

    for g in range(G):
        print(G)
        w_g = W[g * group_size: (g + 1) * group_size]  # (group_size, out_features)
        w_min = w_g.min(dim=0).values
        w_max = w_g.max(dim=0).values

        scale = (w_max - w_min) / (2 ** w_bit - 1)
        scale = torch.clamp(scale, min=1e-5)
        zero = (-w_min / scale).round().to(torch.int32)

        q = ((w_g / scale) + zero).round().clamp(0, 2 ** w_bit - 1)
        dq = (q - zero) * scale
        W_q[g * group_size: (g + 1) * group_size] = dq

        scales[g] = scale
        zeros[g] = zero

    quantized_weight = W_q.t().contiguous()

    # Wrap with WQLinear_GEMM (bias is reused from the original linear layer)
    awq_linear = WQLinear_GEMM.from_linear(
        linear=linear,
        w_bit=w_bit,
        group_size=group_size,
        init_only=False,
        scales=scales.half(),
        zeros=zeros,
    )

    return awq_linear"""

def parallel_worker(i, Xc_cpu, Yc_cpu, E_x, E_y, E_y1, main_device):
    device = torch.device(f"cuda:{i}")
    with torch.no_grad():
        Xc = Xc_cpu.to(device, dtype=torch.float32) - E_x.to(device)
        Yc = Yc_cpu.to(device, dtype=torch.float32) - E_y.to(device)
        Yc1 = (Yc_cpu - Xc_cpu).to(device, dtype=torch.float32) - E_y1.to(device)

        Cyx = (Yc @ Xc.T).to(main_device)
        Cxx = (Xc @ Xc.T).to(main_device)
        Cyy = (Yc @ Yc.T).to(main_device)
        Cyx1 = (Yc1 @ Xc.T).to(main_device)
        print("xxx3")
        del Xc, Yc, Yc1
        torch.cuda.empty_cache()

        return Cyx, Cxx, Cyy, Cyx1

def svd_on_gpu(matrix, device):
    matrix = matrix.to(device)
    U, S, Vh = torch.linalg.svd(matrix, full_matrices=False)
    return U, S, Vh

def process_layer(lyr):
    import time
    start_time = time.time()

    with open("xlayer_objs.pkl", "rb") as f:
        X, Y = pickle.load(f)

    Y = Y + X
    n_total = X.shape[1]
    d_x, d_y = X.shape[0], Y.shape[0]

    n_gpus = torch.cuda.device_count()
    if n_gpus < 1:
        raise RuntimeError("At least one GPU is required.")
    main_device = torch.device("cuda:0")

    print(f"[Layer {lyr}] Using {n_gpus} GPUs")
    print("xxx1")
    chunk_size = n_total // n_gpus
    X_chunks = torch.chunk(X, n_gpus, dim=1)
    Y_chunks = torch.chunk(Y, n_gpus, dim=1)

    E_x = torch.mean(X, dim=1, keepdim=True).to(main_device)
    E_y = torch.mean(Y, dim=1, keepdim=True).to(main_device)

    Cyx_total = torch.zeros((d_y, d_x), dtype=torch.float32, device=main_device)
    Cxx_total = torch.zeros((d_x, d_x), dtype=torch.float32, device=main_device)
    Cyy_total = torch.zeros((d_y, d_y), dtype=torch.float32, device=main_device)
    Y_diff = Y - X
    E_y1 = torch.mean(Y_diff, dim=1, keepdim=True).to(main_device)
    Cyx1_total = torch.zeros((d_y, d_x), dtype=torch.float32, device=main_device)
    
    # Inside process_layer()
    print("xxx2")

    futures = []
    for i in range(n_gpus):
        futures.append(torch.jit.fork(parallel_worker, i, X_chunks[i], Y_chunks[i], E_x, E_y, E_y1, main_device))

    # Initialize accumulators
    Cyx_total = torch.zeros((d_y, d_x), dtype=torch.float32, device=main_device)
    Cxx_total = torch.zeros((d_x, d_x), dtype=torch.float32, device=main_device)
    Cyy_total = torch.zeros((d_y, d_y), dtype=torch.float32, device=main_device)
    Cyx1_total = torch.zeros((d_y, d_x), dtype=torch.float32, device=main_device)

    # Wait and aggregate results
    for f in futures:
        Cyx, Cxx, Cyy, Cyx1 = torch.jit.wait(f)
        Cyx_total += Cyx
        Cxx_total += Cxx
        Cyy_total += Cyy
        Cyx1_total += Cyx1

        del Cyx, Cyx1, Cxx, Cyy
    
    torch.cuda.empty_cache()

    Cyx_total /= (n_total - 1)
    Cxx_total /= (n_total - 1)
    Cyy_total /= (n_total - 1)
    Cyx1_total /= (n_total - 1)

    print("xxx4")
    # GPU SVD
    if n_gpus >= 2:
        # Run in parallel on cuda:0 and cuda:1
        fut_Ux = torch.jit.fork(svd_on_gpu, Cxx_total, torch.device("cuda:0"))
        fut_Uy = torch.jit.fork(svd_on_gpu, Cyy_total, torch.device("cuda:1"))

        Ux, Sx, _ = torch.jit.wait(fut_Ux)
        Uy, Sy, _ = torch.jit.wait(fut_Uy)

        # Ensure both results are on the same device for later ops
        Ux, Sx = Ux.to(main_device), Sx.to(main_device)
        Uy, Sy = Uy.to(main_device), Sy.to(main_device)
    else:
        # Run sequentially on single GPU
        Cxx_total = Cxx_total.to(main_device)
        Cyy_total = Cyy_total.to(main_device)
        Ux, Sx, _ = torch.linalg.svd(Cxx_total, full_matrices=False)
        Uy, Sy, _ = torch.linalg.svd(Cyy_total, full_matrices=False)

    del Cyy_total
    torch.cuda.empty_cache()
    print("xxx5")
    Sx = torch.clamp(Sx, min=1e-9)
    Sy = torch.clamp(Sy, min=1e-9)

    C_XX_inv_sqrt = Ux @ torch.diag(Sx.pow(-0.5)) @ Ux.T
    C_YY_inv_sqrt = Uy @ torch.diag(Sy.pow(-0.5)) @ Uy.T
    corr = C_YY_inv_sqrt @ Cyx_total @ C_XX_inv_sqrt
    del C_YY_inv_sqrt, C_XX_inv_sqrt, Cyx_total, Uy, Ux
    torch.cuda.empty_cache()
    print("xxx6")
    _, S, _ = torch.linalg.svd(corr, full_matrices=False)
    bound = torch.sum(1 - S**2)
    del corr, S, _
    torch.cuda.empty_cache()
    print("xxx7")

    W = Cyx1_total @ torch.linalg.inv(Cxx_total)
    b = E_y1 - W @ E_x
    print("xxx9")
    create_dir(input_path)
    with open(os.path.join(input_path, "layerwise_statsx.txt"), "a") as f:
        f.write(f"for layer {lyr} mse: \n")
        f.write("---------------------------------------------------- \n")
        #f.write(f"sum bound: {(bound * torch.trace(Cyy_total)).item()}\n")
        f.write(f"sum sngs: {bound.item()}\n")
        f.write(f"Elapsed time: {time.time() - start_time:.2f} seconds\n")
        f.write("---------------------------------------------------- \n")

    with open(os.path.join(input_path, f"xlayer_{lyr}_weights.pkl"), "wb") as f:
        pickle.dump([W.cpu(), b.cpu()], f)

    """linear = nn.Linear(W.shape[0], W.shape[1])
    linear.weight = nn.Parameter(W.contiguous())
    linear.bias = nn.Parameter(b.view(-1) if b.ndim > 1 else b)
    linear = linear.to(device)
    quantized_layer = activation_aware_quantize_linear(linear, X.T, Y)

    with open(os.path.join(input_path, f"xlayer_{lyr}_quant.pkl"), "wb") as f:
        pickle.dump([quantized_layer], f)"""
    
    return lyr, bound


if __name__ == "__main__":
    print("Calculating CCA decompositions on GPU...")
    results = [process_layer(layer_index)]
    
    # Save similarities to a file
    with open(input_path + "/similarity_scores.pkl", 'wb') as file:
        pickle.dump(results, file)

    print("similarities\n", results[0])
    print("CCA end file")
