"""Compute-target resolution must degrade to CPU instead of failing."""

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scoop_ai.compute import (  # noqa: E402
    CPU_ANALYSIS_FPS_CEILING,
    ComputeError,
    clamp_analysis_fps,
    configure_torch_threads,
    describe_compute,
    normalise_preference,
    resolve_device,
)


def _fake_torch(*, cuda_available: bool, device_name: str = "NVIDIA GeForce RTX 3050"):
    torch = types.ModuleType("torch")
    torch.__version__ = "2.11.0"
    torch.version = types.SimpleNamespace(cuda="13.0" if cuda_available else None)
    torch.cuda = types.SimpleNamespace(
        is_available=lambda: cuda_available,
        get_device_name=lambda index: device_name,
        get_device_properties=lambda index: types.SimpleNamespace(total_memory=8_589_934_592),
    )
    torch.set_num_threads = mock.Mock()
    return torch


class NormalisePreferenceTests(unittest.TestCase):
    def test_accepts_documented_values(self) -> None:
        for value in ("auto", "cuda", "cpu"):
            self.assertEqual(normalise_preference(value), value)

    def test_accepts_common_synonyms(self) -> None:
        self.assertEqual(normalise_preference("GPU"), "cuda")
        self.assertEqual(normalise_preference("nvidia"), "cuda")
        self.assertEqual(normalise_preference(""), "auto")
        self.assertEqual(normalise_preference(None), "auto")

    def test_rejects_unknown_targets(self) -> None:
        with self.assertRaises(ComputeError):
            normalise_preference("mps")
        with self.assertRaises(ComputeError):
            normalise_preference(7)


class DescribeComputeTests(unittest.TestCase):
    def test_auto_selects_gpu_when_available(self) -> None:
        with mock.patch.dict(sys.modules, {"torch": _fake_torch(cuda_available=True)}):
            capability = describe_compute("auto")
        self.assertEqual(capability.device, "cuda")
        self.assertTrue(capability.accelerated)
        self.assertEqual(capability.device_name, "NVIDIA GeForce RTX 3050")
        self.assertEqual(capability.warnings, ())

    def test_auto_falls_back_to_cpu(self) -> None:
        with mock.patch.dict(sys.modules, {"torch": _fake_torch(cuda_available=False)}):
            capability = describe_compute("auto")
        self.assertEqual(capability.device, "cpu")
        self.assertFalse(capability.accelerated)
        self.assertTrue(any("slower" in warning for warning in capability.warnings))

    def test_cpu_preference_is_honoured_on_a_gpu_machine(self) -> None:
        with mock.patch.dict(sys.modules, {"torch": _fake_torch(cuda_available=True)}):
            capability = describe_compute("cpu")
        self.assertEqual(capability.device, "cpu")
        self.assertTrue(capability.cuda_available)

    def test_requesting_a_missing_gpu_warns_instead_of_raising(self) -> None:
        with mock.patch.dict(sys.modules, {"torch": _fake_torch(cuda_available=False)}):
            capability = describe_compute("cuda")
        self.assertEqual(capability.device, "cpu")
        self.assertTrue(any("GPU was requested" in warning for warning in capability.warnings))

    def test_missing_torch_reports_an_error_without_raising(self) -> None:
        def explode(name, *args, **kwargs):
            if name == "torch":
                raise ImportError("No module named 'torch'")
            return original(name, *args, **kwargs)

        original = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__
        with mock.patch("builtins.__import__", side_effect=explode):
            capability = describe_compute("auto")
        self.assertEqual(capability.device, "cpu")
        self.assertIsNotNone(capability.torch_error)
        self.assertEqual(capability.headline, "AI runtime unavailable")

    def test_payload_is_json_safe(self) -> None:
        import json

        with mock.patch.dict(sys.modules, {"torch": _fake_torch(cuda_available=True)}):
            payload = describe_compute("auto").as_payload()
        self.assertEqual(json.loads(json.dumps(payload))["device"], "cuda")

    def test_resolve_device_matches_description(self) -> None:
        with mock.patch.dict(sys.modules, {"torch": _fake_torch(cuda_available=True)}):
            self.assertEqual(resolve_device("auto"), "cuda")
            self.assertEqual(resolve_device("cpu"), "cpu")


class ThreadAndRateTests(unittest.TestCase):
    def test_gpu_inference_does_not_touch_thread_limits(self) -> None:
        torch = _fake_torch(cuda_available=True)
        with mock.patch.dict(sys.modules, {"torch": torch}):
            self.assertEqual(configure_torch_threads("cuda"), 0)
        torch.set_num_threads.assert_not_called()

    def test_cpu_inference_reserves_a_core(self) -> None:
        torch = _fake_torch(cuda_available=False)
        with mock.patch.dict(sys.modules, {"torch": torch}), \
                mock.patch("scoop_ai.compute._logical_cpus", return_value=8):
            self.assertEqual(configure_torch_threads("cpu"), 7)
        torch.set_num_threads.assert_called_once_with(7)

    def test_single_core_machines_still_get_one_thread(self) -> None:
        torch = _fake_torch(cuda_available=False)
        with mock.patch.dict(sys.modules, {"torch": torch}), \
                mock.patch("scoop_ai.compute._logical_cpus", return_value=1):
            self.assertEqual(configure_torch_threads("cpu"), 1)

    def test_cpu_analysis_rate_is_capped(self) -> None:
        self.assertEqual(clamp_analysis_fps(6.0, "cpu"), CPU_ANALYSIS_FPS_CEILING)

    def test_a_rate_below_the_ceiling_is_left_alone(self) -> None:
        self.assertEqual(clamp_analysis_fps(1.0, "cpu"), 1.0)
        self.assertEqual(clamp_analysis_fps(6.0, "cuda"), 6.0)


if __name__ == "__main__":
    unittest.main()
