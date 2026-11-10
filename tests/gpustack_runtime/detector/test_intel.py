from __future__ import annotations as __future_annotations__

import json

from gpustack_runtime.detector import ManufacturerEnum, intel


def _fake_xpu_smi(*args):
    if args == ("discovery", "-j"):
        return json.dumps(
            {
                "device_list": [
                    {
                        "device_id": 0,
                        "device_name": "Arc",
                        "device_type": "GPU",
                        "device_function_type": "physical",
                        "pci_bdf_address": "0000:2b:00.0",
                        "pci_device_id": "0xe211",
                    },
                    {
                        "device_id": 1,
                        "device_name": "Arc VF",
                        "device_type": "GPU",
                        "device_function_type": "virtual",
                        "pci_bdf_address": "0000:2b:00.1",
                        "pci_device_id": "0xe211",
                    },
                ],
            },
        )
    if args[0] == "discovery":
        return json.dumps(
            {"memory_physical_size_byte": "1048576", "driver_version": "d"},
        )
    return (
        "Timestamp, DeviceId, Average % utilization of all GPU Engines, "
        "GPU Power (W), GPU Memory Used (MiB)\n"
        "03:00:33.934, 0, 50.00, 10.0, 2.0\n"
    )


def test_bdf_to_cdi_uid():
    assert intel.bdf_to_cdi_uid("0000:2b:00.0", "0xe211") == "0000-2b-00-0-0xe211"


def test_collect_from_xpu_smi(monkeypatch):
    monkeypatch.setattr(intel, "_run_xpu_smi", _fake_xpu_smi)
    cards = intel.collect_from_xpu_smi()
    assert list(cards) == ["0000:2b:00.0"]  # the virtual function is skipped
    card = cards["0000:2b:00.0"]
    assert card["uid"] == "0000-2b-00-0-0xe211"
    assert card["mem_total"] == 1048576
    assert card["mem_used"] == 2 * 1024 * 1024
    assert card["util"] == 0.5
    assert card["power"] == 10.0


def test_detect(monkeypatch):
    monkeypatch.setattr(intel, "_run_xpu_smi", _fake_xpu_smi)
    monkeypatch.setattr(intel.shutil, "which", lambda _: "/usr/bin/xpu-smi")
    intel.IntelDetector.is_supported.cache_clear()
    det = intel.IntelDetector()
    devs = det.detect(usage=True)
    intel.IntelDetector.is_supported.cache_clear()
    assert len(devs) == 1
    dev = devs[0]
    assert dev.manufacturer == ManufacturerEnum.INTEL
    assert dev.uuid == "0000-2b-00-0-0xe211"
    assert dev.memory == 1
    assert dev.memory_used == 2
    assert dev.appendix["pci_bdf"] == "0000:2b:00.0"


def _fake_xpu_smi_v2(*args):
    # Captured from xpu-smi 2.1 inside a container with two cards injected.
    if args == ("discovery", "-j"):
        return json.dumps(
            {
                "device_list": [
                    {
                        "device_id": 0,
                        "device_name": "Intel(R) Arc(TM) Pro B60 Graphics",
                        "device_type": "Discrete GPU",
                        "device_function_type": "physical",
                        "pci_bdf_address": "0000:00:00.0",
                        "pci_device_id": "0xe211",
                        "uuid": "00000000-0000-003c-0000-0000e2118086",
                    },
                    {
                        "device_id": 1,
                        "device_name": "Intel(R) Arc(TM) Pro B60 Graphics",
                        "device_type": "Discrete GPU",
                        "device_function_type": "physical",
                        "pci_bdf_address": "0000:00:00.0",
                        "pci_device_id": "0xe211",
                        "uuid": "00000000-0000-0040-0000-0000e2118086",
                    },
                    {  # phantom entry
                        "device_id": 2,
                        "pci_bdf_address": "\x1e",
                        "pci_device_id": "",
                        "uuid": "",
                    },
                ],
            },
        )
    if args[0] == "discovery":
        return json.dumps({"memory_physical_size_byte": 25669140480})
    if "--metrics" in args:
        return (
            "Timestamp, DeviceId, utilization.memory (%), memory.total (MiB), "
            "memory.used (MiB), utilization.gpu (%), power.draw (W), "
            "temperature.gpu (C)\n"
            "04:21:47.893, 0, 43.49, 24480, 10646.5, N/A, 41.72, 25.00\n"
            "04:21:47.893, 1, 0.17, 24480, 41.0, N/A, 42.02, 27.00\n"
        )
    raise AssertionError(args)


def test_collect_from_xpu_smi_v2(monkeypatch):
    monkeypatch.setattr(intel, "_run_xpu_smi", _fake_xpu_smi_v2)
    cards = intel.collect_from_xpu_smi()
    assert sorted(c["uid"] for c in cards.values()) == [
        "0000-3c-00-0-0xe211",
        "0000-40-00-0-0xe211",
    ]
    card = cards["0000:3c:00.0"]
    assert card["mem_used"] == 10646.5 * 1024 * 1024
    assert card["power"] == 41.72
    assert card["temp"] == 25.0
