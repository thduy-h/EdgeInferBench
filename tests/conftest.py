import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import pytest


@pytest.fixture(scope="session")
def exported(tmp_path_factory):
    from edge_infer_bench.conversion import export_onnx

    directory = tmp_path_factory.mktemp("export")
    target = directory / "tiny.onnx"
    metadata = export_onnx("tiny", target, threads=2)
    return target, directory / "tiny_reference", metadata
