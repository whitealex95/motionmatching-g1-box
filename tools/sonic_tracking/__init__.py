"""sonic_tracking -- run NVIDIA's SONIC G1 tracking policy in MuJoCo.

Vendored from the GenoViewPython-MotionMatching project (a pure-Python port
of the official C++ deploy stack in GR00T-WholeBodyControl); only the three
modules this repo needs are kept:

  params.py     -- joint orderings, PD gains, default pose, asset paths.
  rotations.py  -- wxyz quaternion helpers.
  policy.py     -- SonicPolicy: observation pipeline + ONNX inference.

The ONNX checkpoints and the G1 scene live in this repo's assets/sonic/.
"""
