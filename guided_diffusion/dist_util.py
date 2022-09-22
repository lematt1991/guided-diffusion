"""
Helpers for distributed training.
"""

import io
import os
import socket

import blobfile as bf
# from mpi4py import MPI
import torch as th
import torch.distributed as dist
from pytorch_lightning.plugins.environments.slurm_environment import SLURMEnvironment
import torch

# Change this to reflect your cluster layout.
# The GPU for a given rank is (rank % GPUS_PER_NODE).
GPUS_PER_NODE = 8

SETUP_RETRY_COUNT = 3


def get_world_size():
    env = SLURMEnvironment()
    if env.detect():
        return env.world_size()
    return torch.cuda.device_count()

def get_rank():
    env = SLURMEnvironment()
    if env.detect():
        return env.global_rank()
    return int(os.environ["GUIDED_DIFFUSION_RANK"])

def setup_dist():
    """
    Setup a distributed process group.
    """
    if dist.is_initialized():
        return

    env = SLURMEnvironment()

    backend = "nccl"
    local_rank = get_rank() % GPUS_PER_NODE
    os.environ["MASTER_ADDR"] = env.main_address
    os.environ["RANK"] = str(get_rank())
    os.environ["WORLD_SIZE"] = str(get_world_size())
    os.environ["MASTER_PORT"] = str(env.main_port)
    device = torch.device(f'cuda:{local_rank}')
    print(f"Setting device: {device}")
    torch.cuda.set_device(device)
    print(f"About to initialize with rank {get_rank()}")
    dist.init_process_group(backend=backend, init_method="env://")
    print(f"Initialized with rank {get_rank()}")



def dev():
    """
    Get the device to use for torch.distributed.
    """
    if th.cuda.is_available():
        return th.device(f"cuda")
    return th.device("cpu")


def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file without redundant fetches across MPI ranks.
    """
    with bf.BlobFile(path, "rb") as f:
        data = f.read()
    return th.load(io.BytesIO(data), **kwargs)


def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    for p in params:
        with th.no_grad():
            dist.broadcast(p, 0)


def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()
