import numpy as np

from edge_infer_bench.backends import normalize_outputs
from edge_infer_bench.validation import compare_outputs


def test_tolerance_is_relative_to_reference():
    reference = [np.array([0.0, 10.0], dtype=np.float32)]
    assert compare_outputs(reference, [reference[0] + 1e-5], 1e-4, 1e-3)["passed"]
    failed = compare_outputs(reference, [reference[0] + 0.1], 1e-4, 1e-3)
    assert not failed["passed"]
    assert failed["outputs"][0]["mismatched_elements"] == 2
    assert np.isfinite(failed["outputs"][0]["max_relative_error"])


def test_shape_count_dtype_and_nonfinite_failures():
    a = np.ones((2, 3), dtype=np.float32)
    for other in [
        [],
        [a, a],
        [a.reshape(3, 2)],
        [a.astype(np.float64)],
        [a * np.nan],
        [a * np.inf],
    ]:
        assert not compare_outputs([a], other, 1e-4, 1e-3)["passed"]
    assert not compare_outputs([], [], 0, 0)["passed"]
    assert not compare_outputs([np.array([])], [np.array([])], 0, 0)["passed"]
    assert compare_outputs([a], [a], 0, 0)["outputs"][0]["max_absolute_error"] == 0


def test_nested_output_normalization():
    a, b = np.ones(3), np.zeros(2)
    outputs = normalize_outputs({"a": a, "b": (b, [a])})
    assert len(outputs) == 3
    assert np.array_equal(outputs[1], b)


def test_complex_components_are_compared():
    reference = [np.array([1 + 1j], dtype=np.complex64)]
    target = [np.array([1 + 2j], dtype=np.complex64)]
    result = compare_outputs(reference, target, 1e-4, 1e-3)
    assert not result["passed"]
    assert result["outputs"][0]["max_absolute_error"] == 1.0
