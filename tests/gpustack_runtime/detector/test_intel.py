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
