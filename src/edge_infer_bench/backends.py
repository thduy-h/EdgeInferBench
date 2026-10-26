"""Synchronous CPU adapters; optional runtimes are imported only when requested."""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path
from time import perf_counter_ns
from typing import Any

import numpy as np

from edge_infer_bench.artifacts import input_spec, load_metadata
from edge_infer_bench.errors import BackendUnavailable, EdgeBenchError


def normalize_outputs(value: Any) -> list[np.ndarray]:
    """Flatten tuples/lists and insertion-ordered dictionaries to positional tensors."""
    if isinstance(value, Mapping):
        return [a for v in value.values() for a in normalize_outputs(v)]
    if isinstance(value, (tuple, list)):
        return [a for v in value for a in normalize_outputs(v)]
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    result = np.asarray(value)
    if result.dtype.kind not in "biufc":
        raise EdgeBenchError(f"Unsupported output dtype: {result.dtype}")
    return [result]


class InferenceBackend(ABC):
    name: str
    model_metadata: dict[str, Any]
    initialization_ms: float

    @abstractmethod
    def load(self, model_path: Path, cpu_threads: int) -> None: ...

    def prepare_input(self, data: np.ndarray) -> Any:
        return np.ascontiguousarray(data, dtype=np.float32)

    @abstractmethod
    def infer(self, data: Any) -> list[np.ndarray]: ...

    @abstractmethod
    def metadata(self) -> dict[str, Any]: ...

    @abstractmethod
    def close(self) -> None: ...


class PyTorchBackend(InferenceBackend):
    name = "pytorch"

    def load(self, model_path: Path, cpu_threads: int) -> None:
        import torch

        from edge_infer_bench.models import load_reference

        start = perf_counter_ns()
        torch.set_num_threads(cpu_threads)
        if torch.get_num_interop_threads() != 1:
            try:
                torch.set_num_interop_threads(1)
            except RuntimeError:
                pass  # PyTorch forbids changes after parallel work; record actual value below.
        torch.use_deterministic_algorithms(True)
        self.model, self.model_metadata = load_reference(model_path)
        self.settings = {
            "version": torch.__version__,
            "device": "CPU",
            "precision": "FP32",
            "intra_op_threads": torch.get_num_threads(),
            "inter_op_threads": torch.get_num_interop_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "mkldnn_enabled": torch.backends.mkldnn.enabled,
        }
        self.initialization_ms = (perf_counter_ns() - start) / 1e6

    def prepare_input(self, data: np.ndarray) -> Any:
        import torch

        return torch.from_numpy(super().prepare_input(data))

    def infer(self, data: Any) -> list[np.ndarray]:
        import torch

        with torch.inference_mode():
            return normalize_outputs(self.model(data))

    def metadata(self) -> dict[str, Any]:
        return self.settings

    def close(self) -> None:
        self.model = None


class ONNXRuntimeBackend(InferenceBackend):
    name = "onnxruntime"

    def load(self, model_path: Path, cpu_threads: int) -> None:
        try:
            import onnx
            import onnxruntime as ort
        except (ImportError, OSError) as exc:
            raise BackendUnavailable(f"ONNX Runtime unavailable: {exc}") from exc
        start = perf_counter_ns()
        self.model_metadata = load_metadata(model_path)
        onnx.checker.check_model(str(model_path), full_check=True)
        if "CPUExecutionProvider" not in ort.get_available_providers():
            raise BackendUnavailable("ONNX Runtime has no CPUExecutionProvider")
        options = ort.SessionOptions()
        options.intra_op_num_threads = cpu_threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        self.session = ort.InferenceSession(
            str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        inputs = self.session.get_inputs()
        if len(inputs) != 1 or inputs[0].type != "tensor(float)":
            raise EdgeBenchError("v0.1 requires one float32 tensor input")
        self.input_name = inputs[0].name
        self.settings = {
            "version": ort.__version__,
            "device": "CPU",
            "precision": "FP32",
            "providers": self.session.get_providers(),
            "intra_op_threads": cpu_threads,
            "inter_op_threads": 1,
            "execution_mode": "sequential",
            "graph_optimization": "all",
            "allow_spinning": False,
        }
        self.initialization_ms = (perf_counter_ns() - start) / 1e6

    def infer(self, data: Any) -> list[np.ndarray]:
        return normalize_outputs(self.session.run(None, {self.input_name: data}))

    def metadata(self) -> dict[str, Any]:
        return self.settings

    def close(self) -> None:
        self.session = None


class OpenVINOBackend(InferenceBackend):
    name = "openvino"

    def load(self, model_path: Path, cpu_threads: int) -> None:
        try:
            import openvino as ov
        except (ImportError, OSError) as exc:
            raise BackendUnavailable(
                f"OpenVINO unavailable: {exc}. Install edge-infer-bench[openvino]. "
                "AMD CPU compatibility must be established by actual execution."
            ) from exc
        start = perf_counter_ns()
        self.model_metadata = load_metadata(model_path)
        try:
            self.core = ov.Core()
            if "CPU" not in self.core.available_devices:
                raise BackendUnavailable(f"No OpenVINO CPU device: {self.core.available_devices}")
            self.compiled = self.core.compile_model(
                str(model_path),
                "CPU",
                {
                    "INFERENCE_NUM_THREADS": cpu_threads,
                    "NUM_STREAMS": "1",
                    "PERFORMANCE_HINT": "LATENCY",
                    "INFERENCE_PRECISION_HINT": ov.Type.f32,
                },
            )
            self.request = self.compiled.create_infer_request()
            self.settings = {
                "version": ov.__version__,
                "device": "CPU",
                "precision": "FP32",
                "intra_op_threads": int(self.compiled.get_property("INFERENCE_NUM_THREADS")),
                "num_streams": str(self.compiled.get_property("NUM_STREAMS")),
                "performance_hint": str(self.compiled.get_property("PERFORMANCE_HINT")),
                "inference_precision_hint": self.compiled.get_property(
                    "INFERENCE_PRECISION_HINT"
                ).get_type_name(),
            }
        except BackendUnavailable:
            raise
        except RuntimeError as exc:
            raise BackendUnavailable(f"OpenVINO CPU compilation failed: {exc}") from exc
        self.initialization_ms = (perf_counter_ns() - start) / 1e6

    def infer(self, data: Any) -> list[np.ndarray]:
        try:
            self.request.infer({self.compiled.input(0): data})
            return [
                self.request.get_output_tensor(i).data.copy()
                for i in range(len(self.compiled.outputs))
            ]
        except RuntimeError as exc:
            raise EdgeBenchError(f"OpenVINO CPU inference failed: {exc}") from exc

    def metadata(self) -> dict[str, Any]:
        return self.settings

    def close(self) -> None:
        self.request = self.compiled = self.core = None


BACKENDS = {cls.name: cls for cls in (PyTorchBackend, ONNXRuntimeBackend, OpenVINOBackend)}


def create_backend(name: str, path: Path, threads: int) -> InferenceBackend:
    if name not in BACKENDS:
        raise ValueError(f"Unknown backend {name!r}; choose from {', '.join(BACKENDS)}")
    if threads < 1:
        raise ValueError("Threads must be positive")
    adapter = BACKENDS[name]()
    try:
        adapter.load(path, threads)
        input_spec(adapter.model_metadata)  # Reject unsupported contracts before inference.
    except Exception:
        adapter.close()
        raise
    return adapter
