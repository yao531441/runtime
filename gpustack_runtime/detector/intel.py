from __future__ import annotations as __future_annotations__

import logging
import re
import urllib.request
from functools import lru_cache

from .. import envs
from ..logging import debug_log_exception
from .__types__ import (
    Detector,
    Device,
    DeviceMemoryStatusEnum,
    Devices,
    ManufacturerEnum,
    Topology,
    merge_devices_usage,
)
from .__utils__ import (
    byte_to_mebibyte,
    get_numa_node_by_bdf,
    get_utilization,
    map_numa_node_to_cpu_affinity,
)

logger = logging.getLogger(__name__)

_SAMPLE_RE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)"
    r"(?:\{(?P<labels>.*)\})?\s+(?P<value>\S+)(?:\s+\d+)?\s*$",
)
_LABEL_RE = re.compile(r'(?P<k>[a-zA-Z_][a-zA-Z0-9_]*)="(?P<v>(?:[^"\\]|\\.)*)"')

_XPUMD_FETCH_TIMEOUT = 5


def parse_prometheus_text(text: str) -> list[tuple[str, dict[str, str], float]]:
    """
    Parse the Prometheus text exposition format into
    (metric name, labels, value) samples; comments and bad lines are skipped.
    """
    samples: list[tuple[str, dict[str, str], float]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = _SAMPLE_RE.match(line)
        if not m:
            continue
        try:
            value = float(m.group("value"))
        except ValueError:
            continue
        labels = {
            lm.group("k"): lm.group("v")
            for lm in _LABEL_RE.finditer(m.group("labels") or "")
        }
        samples.append((m.group("name"), labels, value))
    return samples


def _discover_xpumd_url(service: str, node_name: str) -> str | None:
    """
    Locate the xpumd Pod of this node through the EndpointSlices of
    `<namespace>/<service>` (xpumd is not hostNetwork): the ready endpoint on
    `node_name`. Needs `endpointslices` get/list permission.
    """
    import kubernetes  # noqa: PLC0415

    namespace, _, name = service.partition("/")
    try:
        kubernetes.config.load_incluster_config()
    except kubernetes.config.config_exception.ConfigException:
        kubernetes.config.load_kube_config()
    api = kubernetes.client.DiscoveryV1Api()
    slices = api.list_namespaced_endpoint_slice(
        namespace,
        label_selector=f"kubernetes.io/service-name={name}",
    )
    for sl in slices.items:
        port = next(
            (p.port for p in (sl.ports or []) if p.name in (None, "http")),
            8080,
        )
        for ep in sl.endpoints or []:
            if ep.node_name == node_name and ep.addresses:
                if ep.conditions and ep.conditions.ready is False:
                    continue
                return f"http://{ep.addresses[0]}:{port}/metrics"
    return None


def _xpumd_url() -> str | None:
    if envs.GPUSTACK_RUNTIME_DETECT_INTEL_XPUMD_URL:
        return envs.GPUSTACK_RUNTIME_DETECT_INTEL_XPUMD_URL
    service = envs.GPUSTACK_RUNTIME_DETECT_INTEL_XPUMD_SERVICE
    node = envs.GPUSTACK_RUNTIME_KUBERNETES_NODE_NAME
    if service and node:
        try:
            return _discover_xpumd_url(service, node)
        except Exception:
            debug_log_exception(logger, "Failed to discover xpumd endpoint")
    return None


def _device_uid(card: dict) -> str:
    """
    Device UID as named by the Intel CDI specs (`intel.com/gpu`), i.e.
    `<domain>-<bus>-<device>-<function>-<pci device id>`, e.g.
    `0000-2b-00-0-0xe211`; it lets the same name be used for the CDI
    device request. Falls back to the xpumd `hw_id`, then the BDF.
    """
    bdf = card["bdf"]
    dev_id = (card.get("info") or {}).get("pci_device_id")
    if dev_id and re.fullmatch(r"[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]", bdf):
        dev_id = dev_id.lower().removeprefix("0x")
        return f"{bdf.replace(':', '-').replace('.', '-')}-0x{dev_id}"
    return card.get("hw_id") or bdf


def _fetch(url: str) -> str:
    with urllib.request.urlopen(url, timeout=_XPUMD_FETCH_TIMEOUT) as resp:  # noqa: S310
        return resp.read().decode("utf-8", errors="ignore")


class IntelDetector(Detector):
    """
    Detect Intel XPUs (e.g. Arc Pro B60) through the node-local xpumd
    Prometheus endpoint, set by `GPUSTACK_RUNTIME_DETECT_INTEL_XPUMD_URL`.
    """

    @staticmethod
    @lru_cache(maxsize=1)
    def is_supported() -> bool:
        """
        Check if the Intel detector is supported, i.e. enabled and the
        xpumd endpoint answers with at least one GPU.
        """
        if envs.GPUSTACK_RUNTIME_DETECT.lower() not in ("auto", "intel"):
            logger.debug("Intel detection is disabled by environment variable")
            return False
        if not _xpumd_url():
            logger.debug("Intel xpumd endpoint is not available")
            return False
        try:
            return bool(IntelDetector._collect())
        except Exception:
            debug_log_exception(logger, "Failed to query Intel xpumd")
            return False

    @staticmethod
    def _collect() -> dict[str, dict]:
        """
        Query xpumd and group the samples by `pci_bdf`.
        """
        url = _xpumd_url()
        if not url:
            return {}
        text = _fetch(url)
        cards: dict[str, dict] = {}
        for name, labels, value in parse_prometheus_text(text):
            bdf = labels.get("pci_bdf")
            if not bdf:
                continue
            card = cards.setdefault(bdf, {"bdf": bdf})
            if name == "hw_gpu_info":
                card["info"] = labels
            elif name == "hw_memory_size_bytes":
                if labels.get("hw_memory_location") == "device":
                    card["mem_total"] = value
            elif name == "hw_memory_usage_bytes":
                if labels.get("hw_memory_location") == "device":
                    card["mem_used"] = value
            elif name == "hw_gpu_utilization_ratio":
                if labels.get("hw_gpu_task") == "all":
                    card["util"] = value
            elif name == "hw_temperature_celsius":
                if labels.get("hw_sensor_location") == "gpu":
                    card["temp"] = value
            elif name == "hw_power_watts":
                if labels.get("hw_sensor_location") == "card":
                    card["power"] = value
            elif name == "hw_status":
                if value >= 1 and labels.get("hw_state") == "reset_needed":
                    card["reset_needed"] = True
            if "hw_id" in labels:
                card.setdefault("hw_id", labels["hw_id"])
        # Only real GPUs: those with device memory or gpu info.
        return {
            bdf: c for bdf, c in cards.items() if "info" in c or "mem_total" in c
        }

    def __init__(self):
        super().__init__(ManufacturerEnum.INTEL)

    @staticmethod
    def _sorted(cards: dict[str, dict]) -> list[dict]:
        # Stable index: sorted by PCI BDF.
        return [cards[k] for k in sorted(cards)]

    @staticmethod
    def _status(card: dict) -> DeviceMemoryStatusEnum:
        if card.get("reset_needed"):
            return DeviceMemoryStatusEnum.UNHEALTHY
        return DeviceMemoryStatusEnum.HEALTHY

    def detect_info(self) -> Devices | None:
        """
        Detect Intel XPU inventory, without usage metrics.
        """
        if not self.is_supported():
            return None

        ret: Devices = []
        try:
            cards = self._collect()
            for idx, card in enumerate(self._sorted(cards)):
                info = card.get("info", {})
                bdf = card["bdf"]
                appendix = {"pci_bdf": bdf}
                for src, dst in (
                    ("hw_vendor", "vendor"),
                    ("hw_model", "model"),
                    ("hw_serial_number", "serial_number"),
                    ("hw_firmware_version", "firmware_version"),
                ):
                    if info.get(src):
                        appendix[dst] = info[src]
                numa = get_numa_node_by_bdf(bdf)
                if numa:
                    appendix["numa"] = numa
                ret.append(
                    Device(
                        manufacturer=self.manufacturer,
                        index=idx,
                        name=info.get("hw_model") or info.get("hw_name") or "Intel XPU",
                        uuid=_device_uid(card),
                        driver_version=info.get("hw_driver_version"),
                        memory=byte_to_mebibyte(int(card.get("mem_total", 0))),
                        memory_status=self._status(card),
                        power=None,
                        appendix=appendix,
                    ),
                )
        except Exception:
            debug_log_exception(logger, "Failed to process Intel devices fetching")
            raise
        return ret

    def detect_usage(self, devices: Devices | None = None) -> Devices | None:
        """
        Fetch the usage of Intel XPUs, merged into the given devices in place.
        """
        if not self.is_supported():
            return None

        if devices is None:
            devices = self.detect_info()
            if devices is None:
                return None

        usages: Devices = []
        try:
            cards = self._collect()
            for card in self._sorted(cards):
                total = byte_to_mebibyte(int(card.get("mem_total", 0)))
                used = byte_to_mebibyte(int(card.get("mem_used", 0)))
                usages.append(
                    Device(
                        uuid=_device_uid(card),
                        cores_utilization=card.get("util", 0) * 100,
                        memory_used=used,
                        memory_utilization=get_utilization(used, total),
                        memory_status=self._status(card),
                        temperature=card.get("temp"),
                        power_used=card.get("power"),
                    ),
                )
        except Exception:
            debug_log_exception(logger, "Failed to process Intel devices usage")
            raise
        merged = merge_devices_usage(devices, usages)
        self._mark_external_allocated(merged)
        return merged

    @staticmethod
    def _external_allocated_bdfs() -> set[str]:
        """
        Cards held by other users' DRA claims on this node; empty when DRA
        occupancy is disabled, no node name is configured or the API fails.
        """
        node = envs.GPUSTACK_RUNTIME_KUBERNETES_NODE_NAME
        if not envs.GPUSTACK_RUNTIME_DETECT_INTEL_DRA_OCCUPANCY or not node:
            return set()
        try:
            import kubernetes  # noqa: PLC0415

            from ..deployer.k8s import dra  # noqa: PLC0415

            try:
                kubernetes.config.load_incluster_config()
            except kubernetes.config.config_exception.ConfigException:
                kubernetes.config.load_kube_config()
            api = kubernetes.client.CustomObjectsApi()
            return dra.get_external_allocated_bdfs(api, node)
        except Exception:
            debug_log_exception(logger, "Failed to read DRA occupancy")
            return set()

    def _mark_external_allocated(self, devices: Devices | None) -> None:
        if not devices:
            return
        held = self._external_allocated_bdfs()
        for dev in devices:
            dev.appendix = dev.appendix or {}
            dev.appendix["external_allocated"] = (
                dev.appendix.get("pci_bdf") in held
            )

    def get_topology(self, devices: Devices | None = None) -> Topology | None:
        """
        Intel topology: only NUMA/CPU affinities; distances stay unknown.
        """
        if devices is None:
            devices = self.detect_info()
            if devices is None:
                return None

        ret = Topology(
            manufacturer=self.manufacturer,
            devices_count=len(devices),
        )
        for i, dev in enumerate(devices):
            ret.devices_numa_affinities[i] = dev.appendix.get("numa", "")
            ret.devices_cpu_affinities[i] = map_numa_node_to_cpu_affinity(
                ret.devices_numa_affinities[i],
            )
        return ret
