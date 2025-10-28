import numpy as np
import json
from pathlib import Path

from cap4d.inference.data.inference_data import CAP4DInferenceDataset


class ReferenceDataset(CAP4DInferenceDataset):
    def __init__(
        self, 
        data_path,
        resolution=512,
        downsample_ratio=8,
    ):
        super().__init__(resolution, downsample_ratio)
        
        self.load_flame_params(data_path)

    def load_flame_params(
        self,
        data_path: Path,
    ):
        flame_dict = dict(np.load(data_path / "fit.npz"))

        with open(data_path / "reference_images.json") as f:
            ref_json = json.load(f)

        ref_list = []
        for cam_name, timestep_id in ref_json:
            cam_id = np.where(flame_dict["camera_order"] == cam_name)[0].item()
            ref_list.append((cam_id, timestep_id))  # cam_id, timestep_id

        flame_list = []
        ref_extr = None
        for cam_id, timestep_id in ref_list:
            # select a single frame (camera, timestep) set from flame_dict
            flame_item = {}

            for key in flame_dict:
                if key in ["expr", "rot", "tra", "eye_rot"]:
                    flame_item[key] = flame_dict[key][[timestep_id]]

                elif key in ["fx", "fy", "cx", "cy", "extr", "resolutions"]:
                    flame_item[key] = flame_dict[key][[cam_id]]

                elif key in ["shape"]:
                    flame_item[key] = flame_dict[key]

            flame_item["timestep_id"] = timestep_id
            cam_dir_path = flame_dict["camera_order"][cam_id]
            flame_item["img_dir_path"] = data_path / "images" / cam_dir_path
            bg_dir_path = data_path / "bg" / cam_dir_path
            if bg_dir_path.exists():
                flame_item["bg_dir_path"] = bg_dir_path

            flame_list.append(flame_item)

            if ref_extr is None:
                ref_extr = flame_item["extr"]


        self.ref_extr = ref_extr[0]
        self.flame_list = flame_list
