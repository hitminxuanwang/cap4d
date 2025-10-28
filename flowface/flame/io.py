import pickle
from typing import Any, List, MutableMapping

# chumpy and scipy are also dependencies because the pkl file requires them
import numpy as np
import torch

import scipy.sparse

def convert_array(
    arr_dict: MutableMapping[str, Any],
    key: str,
    new_dtype: Any = None,
    squeeze=True,
):
    arr = arr_dict[key]
    if callable(getattr(arr, "todense", None)):  # scipy sparse matrix
        arr = arr.todense()
    if new_dtype is None:
        if isinstance(arr.dtype, torch.dtype):
            if arr.dtype.is_floating_point:
                new_dtype = np.float32
        elif np.issubdtype(arr.dtype, np.floating):
            new_dtype = np.float32
        else:
            new_dtype = np.int64
    np_arr = np.array(arr, dtype=new_dtype)
    if squeeze:
        np_arr = np_arr.squeeze()
    arr_dict[key] = np_arr


def load_model_pkl(
    path_model_pkl: str,
):
    '''
    Load a FLAME .pkl file and convert it into numpy array dict
    '''
    model_dict: MutableMapping[str, Any] = pickle.load(
        open(path_model_pkl, "rb"), encoding="latin1"
    )
    keys_to_delete: List[str] = []
    for key, value in model_dict.items():
        if not hasattr(value, "shape"):  # Delete non-array items
            keys_to_delete.append(key)
        elif key == "f":  # use int32 only for index buffers
            convert_array(model_dict, "f", new_dtype=np.int32)
        else:
            convert_array(model_dict, key)
    for key in keys_to_delete:
        del model_dict[key]
    model_dict["kintree_table"][
        0, 0
    ] = -1  # correction for weird 2^32 - 1 value
    return model_dict
