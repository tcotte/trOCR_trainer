import torch


def get_GPU_occupancy(gpu_id: int = 0) -> float:
    """
    Get memory occupancy of the used GPU for training.
    :param gpu_id: id of the GPU used for training model.
    :return: Memory occupancy in percentage.
    """
    if torch.cuda.is_available():
        free_memory, total_memory = torch.cuda.mem_get_info(device=gpu_id)
        return 1 - free_memory / total_memory

    else:
        return 0.0