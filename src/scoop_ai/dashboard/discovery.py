"""Small dependency-free ONVIF/WS-Discovery client for first-run setup."""

from __future__ import annotations

import base64
import hashlib
import os
import socket
import time
import urllib.error
import urllib.request
import uuid
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from urllib.parse import quote, unquote, urlsplit


MULTICAST_ADDRESS = ("239.255.255.250", 3702)
DISCOVERY_NAMESPACE = "http://schemas.xmlsoap.org/ws/2005/04/discovery"
ADDRESSING_NAMESPACE = "http://schemas.xmlsoap.org/ws/2004/08/addressing"
DEVICE_NAMESPACE = "http://www.onvif.org/ver10/device/wsdl"
MEDIA_NAMESPACE = "http://www.onvif.org/ver10/media/wsdl"
SCHEMA_NAMESPACE = "http://www.onvif.org/ver10/schema"
SOAP_NAMESPACE = "http://www.w3.org/2003/05/soap-envelope"
WSSE_NAMESPACE = (
    "http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-secext-1.0.xsd"
)
WSU_NAMESPACE = (
    "http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-utility-1.0.xsd"
)
PASSWORD_DIGEST_TYPE = (
    "http://docs.oasis-open.org/wss/2004/01/"
    "oasis-200401-wss-username-token-profile-1.0#PasswordDigest"
)
NONCE_ENCODING_TYPE = (
    "http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-soap-message-security-1.0#Base64Binary"
)
MAX_ONVIF_RESPONSE_BYTES = 512 * 1024


class OnvifError(RuntimeError):
    """Raised when a camera cannot be interrogated over ONVIF."""


class OnvifAuthError(OnvifError):
    """Raised when a camera rejects the supplied credentials.

    Separate from :class:`OnvifError` so capability probing can fall back for a
    camera that simply does not implement a call, while a bad password fails
    immediately instead of being retried against every endpoint.
    """


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


def _security_header(username: str, password: str) -> str:
    """Build a WS-Security UsernameToken using the digest ONVIF cameras expect.

    The password itself is never transmitted: cameras compare
    ``base64(sha1(nonce + created + password))``.
    """

    nonce = os.urandom(16)
    created = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    digest = hashlib.sha1(nonce + created.encode("utf-8") + password.encode("utf-8")).digest()
    return f'''<s:Header>
 <wsse:Security xmlns:wsse="{WSSE_NAMESPACE}" xmlns:wsu="{WSU_NAMESPACE}">
  <wsse:UsernameToken>
   <wsse:Username>{_escape(username)}</wsse:Username>
   <wsse:Password Type="{PASSWORD_DIGEST_TYPE}">{base64.b64encode(digest).decode()}</wsse:Password>
   <wsse:Nonce EncodingType="{NONCE_ENCODING_TYPE}">{base64.b64encode(nonce).decode()}</wsse:Nonce>
   <wsu:Created>{created}</wsu:Created>
  </wsse:UsernameToken>
 </wsse:Security>
</s:Header>'''


def _escape(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _soap_call(
    url: str,
    body: str,
    *,
    username: str,
    password: str,
    timeout_seconds: float,
) -> ET.Element:
    envelope = (
        f'<?xml version="1.0" encoding="UTF-8"?>'
        f'<s:Envelope xmlns:s="{SOAP_NAMESPACE}">'
        f'{_security_header(username, password)}'
        f"<s:Body>{body}</s:Body></s:Envelope>"
    ).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=envelope,
        headers={"Content-Type": "application/soap+xml; charset=utf-8"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            payload = response.read(MAX_ONVIF_RESPONSE_BYTES)
    except urllib.error.HTTPError as exc:
        detail = exc.read(MAX_ONVIF_RESPONSE_BYTES).decode("utf-8", "replace")
        if exc.code in {401, 403} or "NotAuthorized" in detail or "FailedAuthentication" in detail:
            raise OnvifAuthError(
                "The camera rejected this username and password."
            ) from exc
        raise OnvifError(f"The camera returned an ONVIF error ({exc.code}).") from exc
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        raise OnvifError(
            "The camera's ONVIF service could not be reached. Check that it is on "
            "this network and that ONVIF is enabled."
        ) from exc
    try:
        root = ET.fromstring(payload)
    except ET.ParseError as exc:
        raise OnvifError("The camera sent a response that could not be read.") from exc
    fault = root.findtext(f".//{{{SOAP_NAMESPACE}}}Reason/{{{SOAP_NAMESPACE}}}Text")
    if fault:
        if "NotAuthorized" in fault or "auth" in fault.lower():
            raise OnvifAuthError("The camera rejected this username and password.")
        raise OnvifError(f"The camera reported: {fault.strip()}")
    return root


def _media_service_url(
    service_url: str, *, username: str, password: str, timeout_seconds: float
) -> str:
    """Ask the device which endpoint serves media, falling back to the device one.

    Cameras are inconsistent here: some expose a separate media endpoint, others
    answer media requests on the device service itself.
    """

    try:
        root = _soap_call(
            service_url,
            f'<GetCapabilities xmlns="{DEVICE_NAMESPACE}"><Category>Media</Category></GetCapabilities>',
            username=username,
            password=password,
            timeout_seconds=timeout_seconds,
        )
    except OnvifAuthError:
        # A rejected password will be rejected everywhere; say so now.
        raise
    except OnvifError:
        # Not every camera implements GetCapabilities; media calls often still
        # work against the device endpoint.
        return service_url
    address = root.findtext(
        f".//{{{SCHEMA_NAMESPACE}}}Media/{{{SCHEMA_NAMESPACE}}}XAddr"
    )
    if not address or urlsplit(address).scheme not in {"http", "https"}:
        return service_url
    # Cameras behind NAT sometimes advertise an unreachable address; keep the host
    # discovery actually reached and take only the path.
    discovered = urlsplit(service_url)
    advertised = urlsplit(address)
    if advertised.hostname != discovered.hostname:
        return f"{discovered.scheme}://{discovered.netloc}{advertised.path}"
    return address


def fetch_stream_uri(
    service_url: str,
    *,
    username: str,
    password: str,
    timeout_seconds: float = 8.0,
) -> str:
    """Ask the camera for its own RTSP address instead of guessing a vendor path.

    Returns the URI exactly as the camera reports it, without credentials.
    """

    if urlsplit(service_url).scheme not in {"http", "https"}:
        raise OnvifError("the camera's ONVIF service address is not usable")
    media_url = _media_service_url(
        service_url, username=username, password=password, timeout_seconds=timeout_seconds
    )
    profiles = _soap_call(
        media_url,
        f'<GetProfiles xmlns="{MEDIA_NAMESPACE}"/>',
        username=username,
        password=password,
        timeout_seconds=timeout_seconds,
    )
    tokens = [
        token
        for token in (
            element.get("token")
            for element in profiles.iter(f"{{{MEDIA_NAMESPACE}}}Profiles")
        )
        if token
    ]
    if not tokens:
        raise OnvifError("The camera did not report any video profiles.")
    stream = _soap_call(
        media_url,
        f'<GetStreamUri xmlns="{MEDIA_NAMESPACE}">'
        f"<StreamSetup>"
        f'<Stream xmlns="{SCHEMA_NAMESPACE}">RTP-Unicast</Stream>'
        f'<Transport xmlns="{SCHEMA_NAMESPACE}"><Protocol>RTSP</Protocol></Transport>'
        f"</StreamSetup>"
        f"<ProfileToken>{_escape(tokens[0])}</ProfileToken>"
        f"</GetStreamUri>",
        username=username,
        password=password,
        timeout_seconds=timeout_seconds,
    )
    uri = stream.findtext(f".//{{{SCHEMA_NAMESPACE}}}Uri")
    if not uri or urlsplit(uri.strip()).scheme != "rtsp":
        raise OnvifError("The camera did not report an RTSP stream address.")
    return uri.strip()


def with_credentials(uri: str, username: str, password: str) -> str:
    """Embed credentials in an RTSP URI the way FFmpeg and OpenCV expect them."""

    parts = urlsplit(uri)
    if parts.hostname is None:
        raise OnvifError("the stream address reported by the camera has no host")
    host = f"[{parts.hostname}]" if ":" in parts.hostname else parts.hostname
    authority = f"{quote(username, safe='')}:{quote(password, safe='')}@{host}"
    if parts.port:
        authority = f"{authority}:{parts.port}"
    remainder = parts.path or "/"
    if parts.query:
        remainder = f"{remainder}?{parts.query}"
    return f"{parts.scheme}://{authority}{remainder}"


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
