import faiss, numpy as np

xb = np.load("vectors.npy").astype("float32")
d = xb.shape[1]
nlist = 4096
m = 8
quantizer = faiss.IndexFlatL2(d)
idx = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)


# OPQ pre-rotation for PQ fidelity - optional but recommended
opq = faiss.OPQMatrix(d, min(16, d // 2))
opq.train(xb)
xb_opq = opq.apply_py(xb)

# train the index - coarse + PQ centroids
idx.train(xb_opq)
idx.add(xb_opq)
faiss.write_index(idx, "ivfpq.cpu.index")

# move to GPU and save there
res = faiss.StandardGpuResources()
gpu_idx = faiss.index_cpu_to_gpu(res, 0, idx)
gpu_idx.nprobes = 16

faiss.write_index(faiss.index_gpu_to_cpu(gpu_idx), "ivfpq.opq.index")

# Search knob to sweep
# gpu_idx.nprobes = 8 # fast/less recall
# try [8, 16, 32, 64] chart recall@K vs p95 and pick the knee.

