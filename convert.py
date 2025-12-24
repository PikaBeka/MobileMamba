from functools import partial as _part
from model.mobilemamba.mobilemamba import MobileMamba_S6, replace_batchnorm
import coremltools as ct
import torch
import torch.nn as nn

from model.lib_mamba import vmambanew
from model.lib_mamba.vmambanew import SS2D

import types
from functools import partial
import inspect
import torch

print("Creating model...")
model = MobileMamba_S6(num_classes=1000, mixer_type="minlstm", rnn_expansion=2)
model.eval()
example = torch.randn(1, 3, 224, 224)

# model.load_state_dict(torch.load(
#     "./mobilemamba_s6.pth", map_location="cpu"))

replace_batchnorm(model)

print("Scripting model...")
traced = torch.jit.trace(model, example)

print("Converting to CoreML...")
mlmodel = ct.convert(
    traced,
    inputs=[ct.TensorType(name="input", shape=example.shape)],
    outputs=[ct.TensorType(name="output")],
    debug=True,
)

print("Saving model...")
mlmodel.save("MobileMinLSTM_exp2.mlpackage")
print("✓ Conversion successful!")
