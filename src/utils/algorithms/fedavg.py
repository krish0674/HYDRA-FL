import functools
from collections import OrderedDict


class fedavg:
    """FedAvg server step: element-wise mean of the client state dicts.

    The mean is unweighted (1/N per client), as in the original runs; with the
    HSI-Drive partition used in the paper every client holds 24 or 25 training
    images, so this coincides with the n_i / n weighting of Eq. (1) up to that
    difference. Keys listed in ``exclude_keys`` (the hypernetwork-generated
    layers) are skipped.
    """

    def __init__(self):
        self.algorithm = "FedAvg"

    def aggregate(self, server_state_dict, state_dicts, exclude_keys=()):
        # server_state_dict is unused by FedAvg; kept for a uniform interface.
        exclude = set(exclude_keys)
        result_state_dict = OrderedDict()
        for key in state_dicts[0].keys():
            if key in exclude:
                continue
            current_key_tensors = [state_dict[key] for state_dict in state_dicts]
            current_key_sum = functools.reduce(lambda acc, t: acc + t, current_key_tensors)
            result_state_dict[key] = current_key_sum / len(state_dicts)
        return result_state_dict
