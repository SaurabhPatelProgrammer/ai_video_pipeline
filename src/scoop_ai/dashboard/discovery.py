"""Small dependency-free ONVIF/WS-Discovery client for first-run setup."""

from __future__ import annotations

import socket
import time
import uuid
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from urllib.parse import unquote, urlsplit


MULTICAST_ADDRESS = ("239.255.255.250", 3702)
DISCOVERY_NAMESPACE = "http://schemas.xmlsoap.org/ws/2005/04/discovery"
ADDRESSING_NAMESPACE = "http://schemas.xmlsoap.org/ws/2004/08/addressing"


@dataclass(frozen=True, slots=True)
class DiscoveredCamera:
    device_id: str
    name: str
    host: str
    service_url: str
    scopes: tuple[str, ...]

    def as_payload(self) -> dict[str, object]:
        return asdict(self)


def _probe(message_id: str) -> bytes:
    return f'''<?xml version="1.0" encoding="UTF-8"?>
<e:Envelope xmlns:e="http://www.w3.org/2003/05/soap-envelope"
 xmlns:w="{ADDRESSING_NAMESPACE}" xmlns:d="{DISCOVERY_NAMESPACE}"
 xmlns:dn="http://www.onvif.org/ver10/network/wsdl">
 <e:Header><w:MessageID>uuid:{message_id}</w:MessageID>
 <w:To e:mustUnderstand="true">urn:schemas-xmlsoap-org:ws:2005:04:discovery</w:To>
 <w:Action e:mustUnderstand="true">{DISCOVERY_NAMESPACE}/Probe</w:Action></e:Header>
 <e:Body><d:Probe><d:Types>dn:NetworkVideoTransmitter</d:Types></d:Probe></e:Body>
</e:Envelope>'''.encode("utf-8")


def _scope_name(scopes: tuple[str, ...], host: str) -> str:
    for scope in scopes:
        marker = "/name/"
        if marker in scope:
            value = unquote(scope.split(marker, 1)[1]).replace("_", " ").strip()
            if value:
                return value
    return f"ONVIF camera at {host}"


def parse_probe_matches(payload: bytes) -> list[DiscoveredCamera]:
    """Parse safe display metadata from a WS-Discovery ProbeMatches datagram."""
    try:
        root = ET.fromstring(payload)
    except ET.ParseError:
        return []
    found: list[DiscoveredCamera] = []
    matches = root.findall(f".//{{{DISCOVERY_NAMESPACE}}}ProbeMatch")
    for match in matches:
        address = match.findtext(f".//{{{ADDRESSING_NAMESPACE}}}Address", default="").strip()
        xaddrs = match.findtext(f"{{{DISCOVERY_NAMESPACE}}}XAddrs", default="").split()
        scopes = tuple(match.findtext(f"{{{DISCOVERY_NAMESPACE}}}Scopes", default="").split())
        service_url = next((item for item in xaddrs if urlsplit(item).scheme in {"http", "https"}), "")
        host = urlsplit(service_url).hostname or ""
        if not address or not service_url or not host:
            continue
        found.append(DiscoveredCamera(
            device_id=address,
            name=_scope_name(scopes, host),
            host=host,
            service_url=service_url,
            scopes=scopes,
        ))
    return found


def discover_onvif_cameras(*, timeout_seconds: float = 2.5) -> list[DiscoveredCamera]:
    """Find ONVIF cameras reachable through the machine's default LAN route."""
    if not 0.2 <= timeout_seconds <= 10:
        raise ValueError("discovery timeout must be between 0.2 and 10 seconds")
    probe = _probe(str(uuid.uuid4()))
    cameras: dict[str, DiscoveredCamera] = {}
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP) as client:
        client.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
        client.settimeout(min(0.4, timeout_seconds))
        client.sendto(probe, MULTICAST_ADDRESS)
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            try:
                payload, _address = client.recvfrom(64 * 1024)
            except TimeoutError:
                continue
            except OSError:
                break
            for camera in parse_probe_matches(payload):
                cameras[camera.device_id] = camera
    return sorted(cameras.values(), key=lambda camera: (camera.name.casefold(), camera.host))
