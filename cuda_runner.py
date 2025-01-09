#
# To execute this code, do `modal run main`
#

import sys, os, subprocess
from modal import Image, App, gpu, Mount

app = App("cuda-runner")

def pprint(result, prefix):
    def border(name): return '-'*70 + '\n' + '-'*27 + name + '-'*27 + '\n' + '-'*70
    print(f"{border(f'{prefix} STDOUT')}\n{result.stdout}\n\n\n\n\n{border(f'{prefix} STDERR')}\n{result.stderr}")
    sys.stdout.flush()  # Ensure we see all output


def cuda_mount():
    # upload all files in this directoy to modal GPU
    # needed for cuda files to be uploaded to GPU
    mount = Mount.from_local_dir(".", remote_path="/root/cuda")
    return [mount]


def cuda_image():
    cuda_version = "12.4.0"
    flavor = "devel"
    operating_sys = "ubuntu22.04"
    tag = f"{cuda_version}-{flavor}-{operating_sys}"

    image = (
        Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
        .pip_install(
            "ninja",
            "packaging",
            "wheel",
            "torch",
        )
    )
    return image

# 3 minutes warmpup
@app.function(
    gpu=gpu.A100(),
    image=cuda_image(),
    mounts=cuda_mount(),
)
def run_cuda():

    # Change to the directory containing the CUDA files
    os.chdir("/root/cuda")

    print("Compiling CUDA code...")
    result = subprocess.run(
        ["nvcc", "-v", "main.cu", "-o", "main"],
        capture_output=True,
        text=True
    )
    pprint(result, 'Compilation')

    if result.returncode != 0:
        raise Exception(f"Compilation failed:\n{result.stderr}")

    print("\nRunning CUDA program...")
    result = subprocess.run(["./main"], capture_output=True, text=True)
    pprint(result, 'Execution')
    sys.stdout.flush()
