from __future__ import annotations

from pathlib import Path

import pytest

from gpustack_runtime import envs
from gpustack_runtime.detector import intel
from gpustack_runtime.detector.__types__ import (
    DeviceMemoryStatusEnum,
    ManufacturerEnum,
    manufacturer_to_backend,
)
from gpustack_runtime.detector.intel import IntelDetector

_SAMPLE = Path(__file__).parent / "samples" / "xpumd_metrics_intel_b60.txt"


@pytest.fixture
def det(monkeypatch):
    monkeypatch.setattr(envs, "GPUSTACK_RUNTIME_DETECT", "auto", raising=False)
    monkeypatch.setattr(
        envs,
        "GPUSTACK_RUNTIME_DETECT_INTEL_XPUMD_URL",
        "http://fake/metrics",
        raising=False,
    )
    monkeypatch.setattr(intel, "_fetch", lambda _url: _SAMPLE.read_text())
    IntelDetector.is_supported.cache_clear()
    yield IntelDetector()
    IntelDetector.is_supported.cache_clear()


def test_backend():
    assert manufacturer_to_backend(ManufacturerEnum.INTEL) == "oneapi"


def test_disabled_without_url(monkeypatch):
    monkeypatch.setattr(
        envs, "GPUSTACK_RUNTIME_DETECT_INTEL_XPUMD_URL", None, raising=False
    )
    IntelDetector.is_supported.cache_clear()
    assert IntelDetector.is_supported() is False
    IntelDetector.is_supported.cache_clear()


def test_detect_info(det):
    devs = det.detect_info()
    assert len(devs) == 8
    assert [d.index for d in devs] == list(range(8))
    bdfs = [d.appendix["pci_bdf"] for d in devs]
    assert bdfs == sorted(bdfs)
    d = next(x for x in devs if x.appendix["pci_bdf"] == "0000:b0:00.0")
    assert d.manufacturer == ManufacturerEnum.INTEL
    assert d.uuid == "868011e2-0000-0000-b000-000000000000"
    assert "B60" in d.name
    assert d.memory == 24480  # 25669140480 B
    # reset_needed is exposed with value 0 in the sample: still healthy.
    assert d.memory_status == DeviceMemoryStatusEnum.HEALTHY


def test_detect_usage(det):
    devs = det.detect_usage()
    d = next(x for x in devs if x.appendix["pci_bdf"] == "0000:b0:00.0")
    assert d.memory_used == 22099  # 23172841472 B, floored
    assert 90 < d.memory_utilization < 91
    assert d.temperature == 31
    assert d.cores_utilization == 0
