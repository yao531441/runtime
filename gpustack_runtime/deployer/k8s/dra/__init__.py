from __future__ import annotations as __future_annotations__

import logging
from typing import TYPE_CHECKING, Any

from gpustack_runtime import envs

if TYPE_CHECKING:
    import kubernetes

logger = logging.getLogger(__name__)

DRA_GROUP = "resource.k8s.io"
DRA_VERSION = "v1"

ALLOCATED_ANNOTATION = "device.gpustack.ai/accelerator.allocated"
"""
Pod annotation carrying the accelerators DRA really allocated.
"""


def split_topology_requests(
    count: int,
    group_size: int | None,
    *,
    allow_partial: bool = False,
) -> list[int]:
    """
    Split `count` devices into per-request sizes.

    With a known topology group size g (devices sharing one topology domain,
    e.g. one pcieRoot), a multi-device claim takes whole domains only:
    count/g requests of `g` devices each, every one bound to one domain.
    A single device needs no constraint. A count that is not a multiple of g
    is rejected unless `allow_partial`, which appends a remainder request.

    Raises:
        ValueError: If the count is invalid or not a multiple of g.

    """
    if count < 1:
        msg = f"Invalid device count {count}"
        raise ValueError(msg)
    if count == 1 or not group_size or group_size <= 1:
        return [count]
    whole, rest = divmod(count, group_size)
    if whole == 0 or rest:
        if not allow_partial:
            msg = (
                f"Requested {count} devices is not a multiple of the topology "
                f"domain size {group_size}; whole domains are required "
                "(set allow_partial_domain to relax)"
            )
            raise ValueError(msg)
        return [group_size] * whole + ([rest] if rest else [])
    return [group_size] * whole


def build_claim_template(
    name: str,
    namespace: str,
    labels: dict[str, str],
    count: int,
    *,
    device_class: str | None = None,
    attribute: str | None = None,
    group_size: int | None = None,
    allow_partial: bool | None = None,
) -> dict[str, Any]:
    """
    Build a ResourceClaimTemplate (resource.k8s.io/v1) body.
    """
    device_class = device_class or envs.GPUSTACK_RUNTIME_KUBERNETES_DRA_DEVICE_CLASS
    if attribute is None:
        attribute = envs.GPUSTACK_RUNTIME_KUBERNETES_DRA_TOPOLOGY_ATTRIBUTE
    if allow_partial is None:
        allow_partial = envs.GPUSTACK_RUNTIME_KUBERNETES_DRA_ALLOW_PARTIAL_DOMAIN
    sizes = split_topology_requests(
        count,
        group_size if attribute else None,
        allow_partial=allow_partial,
    )
    requests = []
    constraints = []
    for i, size in enumerate(sizes):
        rname = f"gpu-{i}"
        requests.append(
            {
                "name": rname,
                "exactly": {
                    "deviceClassName": device_class,
                    "allocationMode": "ExactCount",
                    "count": size,
                },
            },
        )
        if attribute and size > 1:
            constraints.append({"requests": [rname], "matchAttribute": attribute})
    spec: dict[str, Any] = {"devices": {"requests": requests}}
    if constraints:
        spec["devices"]["constraints"] = constraints
    return {
        "apiVersion": f"{DRA_GROUP}/{DRA_VERSION}",
        "kind": "ResourceClaimTemplate",
        "metadata": {"name": name, "namespace": namespace, "labels": labels},
        "spec": {"spec": spec},
    }


def _attr_value(attr: dict[str, Any]) -> Any:
    for k in ("string", "int", "bool", "version"):
        if k in attr:
            return attr[k]
    return None


def discover_domains(
    custom_api: kubernetes.client.CustomObjectsApi,
    node_name: str,
    attribute: str,
    driver: str | None = None,
) -> dict[Any, list[str]]:
    """
    Group the devices published for `node_name` by the value of `attribute`,
    e.g. pcieRoot -> device names. Reads ResourceSlices, read-only.
    """
    driver = driver or envs.GPUSTACK_RUNTIME_KUBERNETES_DRA_DRIVER
    slices = custom_api.list_cluster_custom_object(
        DRA_GROUP,
        DRA_VERSION,
        "resourceslices",
        field_selector=f"spec.nodeName={node_name},spec.driver={driver}",
    )
    domains: dict[Any, list[str]] = {}
    for item in slices.get("items", []):
        for dev in item.get("spec", {}).get("devices", []) or []:
            attrs = dev.get("attributes") or {}
            if attribute not in attrs:
                continue
            domains.setdefault(_attr_value(attrs[attribute]), []).append(dev["name"])
    return domains


def resolve_group_size(
    custom_api: kubernetes.client.CustomObjectsApi | None,
    node_name: str | None,
) -> int | None:
    """
    Topology domain size: the env override, else the (uniform) domain size
    read from the node's ResourceSlices; None when unknown or disabled.
    """
    attribute = envs.GPUSTACK_RUNTIME_KUBERNETES_DRA_TOPOLOGY_ATTRIBUTE
    if not attribute:
        return None
    if envs.GPUSTACK_RUNTIME_KUBERNETES_DRA_DOMAIN_SIZE:
        return int(envs.GPUSTACK_RUNTIME_KUBERNETES_DRA_DOMAIN_SIZE)
    if custom_api is None or not node_name:
        return None
    domains = discover_domains(custom_api, node_name, attribute)
    sizes = {len(v) for v in domains.values()}
    if len(sizes) == 1:
        return sizes.pop()
    if sizes:
        msg = f"Non-uniform topology domain sizes {sorted(sizes)} on node {node_name}"
        raise ValueError(msg)
    return None


def get_allocated_devices(
    custom_api: kubernetes.client.CustomObjectsApi,
    core_api: kubernetes.client.CoreV1Api,
    namespace: str,
    pod_name: str,
    node_name: str,
    driver: str | None = None,
) -> list[dict[str, Any]]:
    """
    Resolve what DRA really allocated to a Pod: the Pod's claims
    (`status.resourceClaimStatuses`) -> allocation results -> device
    attributes from the node's ResourceSlices.

    Returns:
        A list of {"name", "pci_bdf"} dicts, sorted by pci_bdf.

    """
    driver = driver or envs.GPUSTACK_RUNTIME_KUBERNETES_DRA_DRIVER
    pod = core_api.read_namespaced_pod(pod_name, namespace)
    claim_names = [
        s.resource_claim_name
        for s in (pod.status.resource_claim_statuses or [])
        if s.resource_claim_name
    ]
    names: set[str] = set()
    for cn in claim_names:
        claim = custom_api.get_namespaced_custom_object(
            DRA_GROUP,
            DRA_VERSION,
            namespace,
            "resourceclaims",
            cn,
        )
        results = (
            (claim.get("status") or {}).get("allocation", {}).get("devices", {})
        ).get("results", [])
        names.update(r["device"] for r in results if r.get("driver") == driver)
    if not names:
        return []
    slices = custom_api.list_cluster_custom_object(
        DRA_GROUP,
        DRA_VERSION,
        "resourceslices",
        field_selector=f"spec.nodeName={node_name},spec.driver={driver}",
    )
    bdf_attr = envs.GPUSTACK_RUNTIME_KUBERNETES_DRA_PCI_ATTRIBUTE
    ret = []
    for item in slices.get("items", []):
        for dev in item.get("spec", {}).get("devices", []) or []:
            if dev["name"] in names:
                bdf = _attr_value((dev.get("attributes") or {}).get(bdf_attr, {}))
                ret.append({"name": dev["name"], "pci_bdf": bdf})
    return sorted(ret, key=lambda d: str(d["pci_bdf"]))


def get_external_allocated_bdfs(
    custom_api: kubernetes.client.CustomObjectsApi,
    node_name: str,
    driver: str | None = None,
) -> set[str]:
    """
    PCI addresses of the node's devices DRA allocated to claims that do not
    belong to this runtime (claims without the workload label), i.e. cards
    other users hold. Read-only.
    """
    driver = driver or envs.GPUSTACK_RUNTIME_KUBERNETES_DRA_DRIVER
    own_label = f"{envs.GPUSTACK_RUNTIME_DEPLOY_LABEL_PREFIX}/workload"
    claims = custom_api.list_cluster_custom_object(
        DRA_GROUP,
        DRA_VERSION,
        "resourceclaims",
    )
    names: set[str] = set()
    for claim in claims.get("items", []):
        if own_label in (claim.get("metadata", {}).get("labels") or {}):
            continue
        results = (
            (claim.get("status") or {}).get("allocation", {}).get("devices", {})
        ).get("results", [])
        names.update(
            r["device"]
            for r in results
            if r.get("driver") == driver
            and (r.get("pool") in (None, node_name) or node_name in str(r.get("pool")))
        )
    if not names:
        return set()
    slices = custom_api.list_cluster_custom_object(
        DRA_GROUP,
        DRA_VERSION,
        "resourceslices",
        field_selector=f"spec.nodeName={node_name},spec.driver={driver}",
    )
    bdf_attr = envs.GPUSTACK_RUNTIME_KUBERNETES_DRA_PCI_ATTRIBUTE
    ret = set()
    for item in slices.get("items", []):
        for dev in item.get("spec", {}).get("devices", []) or []:
            if dev["name"] in names:
                bdf = _attr_value((dev.get("attributes") or {}).get(bdf_attr, {}))
                if bdf:
                    ret.add(str(bdf))
    return ret
