# Installing motionmatching-g1-box

Everything needed for the interactive demo (`run.py`) is in this repo: the G1
model, the motion data, and the code. No sudo, no GPU driver, no CUDA. It needs
conda (Miniconda is enough). Only the optional real-robot bridge needs a second repo.

Tested with Python 3.10 on Ubuntu 22.04 and 24.04.

## 1. System packages

Any Linux desktop with a working OpenGL display. `git` is enough on most
machines. If `import glfw` fails later, install the GLFW shared library:

```bash
sudo apt-get install -y git libglfw3
```

## 2. Conda environment

The env is named `mm-g1-sonic`. The run docs in `motionmatching-g1-deploy` use
that name too.

```bash
git clone https://github.com/whitealex95/motionmatching-g1-box.git
cd motionmatching-g1-box
conda create -n mm-g1-sonic python=3.10 -y
conda activate mm-g1-sonic
pip install -r requirements.txt          # mujoco, glfw, numpy, scipy
python run.py --build-only               # builds data/motion_lib.npz (~1 s)
```

Check:

```bash
python run.py                            # opens the window, WASD to move, Esc to quit
```

The viewer needs a display. On a headless host use a desktop or X-forwarded
session with `MUJOCO_GL=glfw` (the default).

## 3. Extras for the SONIC tracking sims (`sonic_g1_box/`)

`run_kinematic.py`, `run_grasp.py`, `run_track_clips.py` and
`run_interactive.py` run the SONIC policy locally in MuJoCo. They need ONNX
Runtime and PyYAML on top of step 2:

```bash
pip install onnxruntime pyyaml
```

The default `release` checkpoint is in git. The `low_latency` and
`sonic_v1_1` decoders are about 150 MB each and are gitignored:

```bash
bash assets/sonic/policy/fetch_models.sh # only if you want --sonic low_latency / sonic_v1_1
```

`tools/make_medicine_box.py` also needs `pip install pillow`.

## 4. Extras for the real-robot bridge (`sonic_g1_box/run_hardware.py`)

The bridge streams the matcher to the C++ deploy node in
[`motionmatching-g1-deploy`](https://github.com/whitealex95/motionmatching-g1-deploy)
over ZMQ. It needs pyzmq, and msgpack to decode the node's `g1_debug` stream
(the live robot view):

```bash
pip install pyzmq msgpack
python sonic_g1_box/run_hardware.py --help
```

Install and build the deploy node by following `INSTALL.md` in that repo (CUDA,
TensorRT, ONNX Runtime, the planner file). The bridge alone can be tested
against the deploy repo's Python monitor without building anything:

```bash
# terminal 1, in motionmatching-g1-deploy
python tools/zmq_pose_monitor.py --seconds 10
# terminal 2, here
python sonic_g1_box/run_hardware.py --headless 8
```

## Common problems

- **`ModuleNotFoundError: mm_g1`** when running a script in `sonic_g1_box/`.
  Run it from the repo root, for example `python sonic_g1_box/run_hardware.py`,
  or `cd sonic_g1_box` first. Both work, the scripts add the repo root to
  `sys.path`.
- **The cache is stale after editing `mm_g1/config/library.py`.** It rebuilds
  by itself on the next load. Delete `data/motion_lib.npz` to force it.
- **`glfw.GLFWError` or a black window over SSH.** There is no display or no
  GPU-accelerated OpenGL. Run on the desktop, or use `--headless` where a
  script offers it.
- **`Address already in use` on port 5556.** Another bridge is still running.
  Kill it, or pass `--port`.
