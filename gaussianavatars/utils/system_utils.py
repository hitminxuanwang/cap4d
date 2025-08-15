#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from errno import EEXIST
from os import makedirs, path
import os
from pathlib import Path


def mkdir_p(folder_path):
    # Creates a directory. equivalent to using mkdir -p on the command line
    try:
        makedirs(folder_path)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(folder_path):
            pass
        else:
            raise


def searchForMaxIteration(folder):
    files = Path(folder).glob("*.pth")
    saved_iters = [(int(f.stem[6:]), f) for f in files]

    if saved_iters:
        latest_iter, latest_file = max(saved_iters, key=lambda x: x[0])
        return latest_iter, latest_file
    else:
        return None, None
