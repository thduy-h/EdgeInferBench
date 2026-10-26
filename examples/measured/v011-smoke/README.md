# v0.1.1 smoke evidence

This directory retains actual v0.1.1 smoke outputs from the AMD Ryzen 7 4800H
machine. The tiny convolutional model used deterministic random weights, FP32 NCHW
inputs, seed 7 for ONNX Runtime and seed 11 for the OpenVINO conversion example,
three isolated child trials, five warmups, and 30 timed iterations for each batch.
These are packaging and pipeline evidence, not representative model performance.
The OpenVINO files demonstrate successful installation, ONNX-to-IR conversion, CPU
compilation, inference, and numerical validation on this machine for this model.
