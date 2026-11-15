import functools
from collections import OrderedDict


class fedprox:
    """FedProx server step. The proximal term acts on the client side (see
    ``net_lib.train_h_fedprox``); the server step is the same unweighted mean as
    FedAvg. Keys in ``exclude_keys`` are skipped."""

    def __init__(self):
        self.algorithm = "FedProx"

    def aggregate(self, server_state_dict, state_dicts, exclude_keys=()):
        exclude = set(exclude_keys)
        result_state_dict = OrderedDict()
        for key in state_dicts[0].keys():
            if key in exclude:
                continue
            current_key_tensors = [state_dict[key] for state_dict in state_dicts]
            current_key_sum = functools.reduce(lambda acc, t: acc + t, current_key_tensors)
            result_state_dict[key] = current_key_sum / len(state_dicts)
        return result_state_dict
