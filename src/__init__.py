import torch


def get_device():
    """
    Get the device on which the experiments have to be run.
    :return: the device on which the experiments have to be run.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")  # Use the first GPU
    else:
        return torch.device("cpu")