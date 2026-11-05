from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gpustack_runtime.deployer.k8s import dra


@pytest.mark.parametrize(
    "count, g, partial, expected",
    [
        (1, 2, False, [1]),
        (2, 2, False, [2]),
        (4, 2, False, [2, 2]),
        (8, 2, False, [2, 2, 2, 2]),
        (3, 2, True, [2, 1]),
        (1, None, False, [1]),
        (4, None, False, [4]),
    ],
)
def test_split(count, g, partial, expected):
    assert dra.split_topology_requests(count, g, allow_partial=partial) == expected


@pytest.mark.parametrize("count", [3, 5])
def test_split_rejects_partial(count):
    with pytest.raises(ValueError, match="whole domains"):
        dra.split_topology_requests(count, 2)


def test_template_without_attribute_has_no_constraint():
    body = dra.build_claim_template("n", "ns", {}, 4, attribute="", group_size=2)
    devs = body["spec"]["spec"]["devices"]
    assert devs["requests"][0]["exactly"]["count"] == 4
    assert "constraints" not in devs


def test_get_allocated_devices():
    core = MagicMock()
    core.read_namespaced_pod.return_value = SimpleNamespace(
        status=SimpleNamespace(
            resource_claim_statuses=[SimpleNamespace(resource_claim_name="c1")],
        ),
    )
    custom = MagicMock()
    custom.get_namespaced_custom_object.return_value = {
        "status": {
            "allocation": {
                "devices": {
                    "results": [
                        {"driver": "gpu.intel.com", "device": "card-b"},
                        {"driver": "gpu.intel.com", "device": "card-a"},
                        {"driver": "other", "device": "card-x"},
                    ],
                },
            },
        },
    }
    attr = "resource.kubernetes.io/pciBusID"
    custom.list_cluster_custom_object.return_value = {
        "items": [
            {
                "spec": {
                    "devices": [
                        {"name": n, "attributes": {attr: {"string": b}}}
                        for n, b in (
                            ("card-a", "0000:26:00.0"),
                            ("card-b", "0000:37:00.0"),
                            ("card-x", "0000:ff:00.0"),
                        )
                    ],
                },
            },
        ],
    }
    ret = dra.get_allocated_devices(custom, core, "ns", "pod", "node-a")
    assert [d["pci_bdf"] for d in ret] == ["0000:26:00.0", "0000:37:00.0"]


def test_external_allocated_excludes_own_claims(monkeypatch):
    from gpustack_runtime import envs

    own = f"{envs.GPUSTACK_RUNTIME_DEPLOY_LABEL_PREFIX}/workload"

    def alloc(device, labels=None, admin=False):
        result = {"driver": "gpu.intel.com", "device": device}
        if admin:
            result["adminAccess"] = True
        return {
            "metadata": {"labels": labels or {}},
            "status": {
                "allocation": {
                    "devices": {
                        "results": [result],
                    },
                },
            },
        }

    custom = MagicMock()
    attr = "resource.kubernetes.io/pciBusID"

    def lister(group, version, plural, **kw):
        if plural == "resourceclaims":
            return {"items": [alloc("a"), alloc("b", {own: "w"}), alloc("c", admin=True)]}
        return {
            "items": [
                {
                    "spec": {
                        "devices": [
                            {"name": "a", "attributes": {attr: {"string": "A"}}},
                            {"name": "b", "attributes": {attr: {"string": "B"}}},
                            {"name": "c", "attributes": {attr: {"string": "C"}}},
                        ],
                    },
                },
            ],
        }

    custom.list_cluster_custom_object.side_effect = lister
    assert dra.get_external_allocated_bdfs(custom, "node-a") == {"A"}
