import io
from pathlib import Path

import torch


def save_tensor(tensor, name):
    f = io.BytesIO()
    torch.save(tensor, f, _use_new_zipfile_serialization=True)
    with open(name, "wb") as out_f:
        out_f.write(f.getbuffer())


def process_forward_dump(dump_path: Path, output_path: Path):
    output_path.mkdir(exist_ok=True, parents=True)
    data = torch.load(dump_path, weights_only=False)
    arg_names = [
        'sigma', 
        'weight_background', 
        'bg', 
        'anchor_means3D',
        'anchor_scales',
        'anchor_rotations',
        'means3D', 
        'colors_precomp', 
        'opacities', 
        'vi', 
        'scales', 
        'rotations', 
        'scale_modifier', 
        'cov3Ds_precomp', 
        'viewmatrix', 
        'projmatrix', 
        'tanfovx', 
        'tanfovy', 
        'image_height', 
        'image_width', 
        'sh', 
        'sh_degree', 
        'campos', 
        'prefiltered', 
        'debug',
        'ET_grade',
        'if_depth_correct'
    ]
    for tensor, name in zip(data, arg_names):
        save_tensor(tensor, str(output_path / name) + ".pt")
    # save_tensor(torch.tensor([]), str(output_path / 'sh') + ".pt")
    # save_tensor(int(3), str(output_path / 'sh_degree') + ".pt")


def process_backward_dump(dump_path: Path, output_path: Path):
    output_path.mkdir(exist_ok=True, parents=True)
    data = torch.load(dump_path, weights_only=False)
    arg_names = [
        "sigma",
        "render_image",
        "opacities",
        "vi",
        "bg",
        "means3D",
        "radii",
        "colors_precomp",
        "scales",
        "rotations",
        "scale_modifier",
        "cov3Ds_precomp",
        "viewmatrix",
        "projmatrix",
        "tanfovx",
        "tanfovy",
        "grad_out_color",
        "sh",
        "sh_degree",
        "campos",
        "geomBuffer",
        "num_rendered",
        "binningBuffer",
        "imgBuffer",
        "debug",
        "ET_grade",
        "if_depth_correct",
    ]
    for tensor, name in zip(data, arg_names):
        save_tensor(tensor, str(output_path / name) + ".pt")


if __name__ == '__main__':
    global_path = Path("./test_data")
    process_forward_dump("./test_data/snapshot_fw.dump", global_path / "forward_tensors")
    process_backward_dump("./test_data/snapshot_bw.dump",  global_path / "backward_tensors")
