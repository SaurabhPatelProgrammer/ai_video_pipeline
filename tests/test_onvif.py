"""Cameras are authoritative about their own RTSP address; guessing one fails."""

from __future__ import annotations

import base64
import hashlib
import io
import sys
import unittest
import urllib.error
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scoop_ai.dashboard.discovery import (  # noqa: E402
    MEDIA_NAMESPACE,
    SCHEMA_NAMESPACE,
    WSSE_NAMESPACE,
    WSU_NAMESPACE,
    OnvifAuthError,
    OnvifError,
    fetch_stream_uri,
    with_credentials,
)


def _envelope(body: str) -> bytes:
    return (
        '<?xml version="1.0"?>'
        '<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope">'
        f"<s:Body>{body}</s:Body></s:Envelope>"
    ).encode()


CAPABILITIES = _envelope(
    f'<GetCapabilitiesResponse xmlns="http://www.onvif.org/ver10/device/wsdl">'
    f'<Capabilities xmlns="{SCHEMA_NAMESPACE}">'
    f"<Media><XAddr>http://192.168.2.118/onvif/media_service</XAddr></Media>"
    f"</Capabilities></GetCapabilitiesResponse>"
)
PROFILES = _envelope(
    f'<GetProfilesResponse xmlns="{MEDIA_NAMESPACE}">'
    f'<Profiles token="MainStream"><Name>main</Name></Profiles>'
    f'<Profiles token="SubStream"><Name>sub</Name></Profiles>'
    f"</GetProfilesResponse>"
)
STREAM = _envelope(
    f'<GetStreamUriResponse xmlns="{MEDIA_NAMESPACE}">'
    f'<MediaUri xmlns="{SCHEMA_NAMESPACE}">'
    f"<Uri>rtsp://192.168.2.118:554/stream1</Uri>"
    f"</MediaUri></GetStreamUriResponse>"
)


class _Transport:
    """Reply to each SOAP call in turn and record what was sent."""

    def __init__(self, *responses: bytes) -> None:
        self.responses = list(responses)
        self.requests: list[tuple[str, bytes]] = []

    def __call__(self, request, timeout=None):
        self.requests.append((request.full_url, request.data))
        payload = self.responses.pop(0)
        if isinstance(payload, Exception):
            raise payload
        response = mock.MagicMock()
        response.read.return_value = payload
        response.__enter__ = lambda _self: response
        response.__exit__ = lambda *_args: False
        return response


class FetchStreamUriTests(unittest.TestCase):
    def test_asks_the_camera_instead_of_guessing_a_vendor_path(self) -> None:
        transport = _Transport(CAPABILITIES, PROFILES, STREAM)
        with mock.patch("scoop_ai.dashboard.discovery.urllib.request.urlopen", transport):
            uri = fetch_stream_uri(
                "http://192.168.2.118/onvif/device_service",
                username="admin",
                password="secret",
            )
        # The path is the camera's own, not the /stream that a guess would produce.
        self.assertEqual(uri, "rtsp://192.168.2.118:554/stream1")
        self.assertEqual(len(transport.requests), 3)
        self.assertIn(b"GetProfiles", transport.requests[1][1])
        self.assertIn(b"MainStream", transport.requests[2][1])

    def test_sends_a_password_digest_and_never_the_password(self) -> None:
        transport = _Transport(CAPABILITIES, PROFILES, STREAM)
        with mock.patch("scoop_ai.dashboard.discovery.urllib.request.urlopen", transport):
            fetch_stream_uri(
                "http://cam/onvif/device_service", username="admin", password="s3cr3t!"
            )
        sent = transport.requests[0][1]
        self.assertNotIn(b"s3cr3t!", sent)
        root = ET.fromstring(sent)
        nonce = root.findtext(f".//{{{WSSE_NAMESPACE}}}Nonce")
        created = root.findtext(f".//{{{WSU_NAMESPACE}}}Created")
        digest = root.findtext(f".//{{{WSSE_NAMESPACE}}}Password")
        expected = base64.b64encode(
            hashlib.sha1(
                base64.b64decode(nonce) + created.encode() + b"s3cr3t!"
            ).digest()
        ).decode()
        self.assertEqual(digest, expected)

    def test_media_requests_go_to_the_host_discovery_reached(self) -> None:
        """A camera advertising an unroutable address must not strand setup."""

        capabilities = _envelope(
            f'<GetCapabilitiesResponse xmlns="http://www.onvif.org/ver10/device/wsdl">'
            f'<Capabilities xmlns="{SCHEMA_NAMESPACE}">'
            f"<Media><XAddr>http://10.0.0.5/onvif/media</XAddr></Media>"
            f"</Capabilities></GetCapabilitiesResponse>"
        )
        transport = _Transport(capabilities, PROFILES, STREAM)
        with mock.patch("scoop_ai.dashboard.discovery.urllib.request.urlopen", transport):
            fetch_stream_uri(
                "http://192.168.2.118/onvif/device_service",
                username="admin",
                password="x",
            )
        self.assertEqual(transport.requests[1][0], "http://192.168.2.118/onvif/media")

    def test_falls_back_to_the_device_endpoint_when_capabilities_fail(self) -> None:
        transport = _Transport(
            urllib.error.HTTPError("u", 500, "err", {}, io.BytesIO(b"boom")),
            PROFILES,
            STREAM,
        )
        with mock.patch("scoop_ai.dashboard.discovery.urllib.request.urlopen", transport):
            uri = fetch_stream_uri(
                "http://cam/onvif/device_service", username="admin", password="x"
            )
        self.assertEqual(uri, "rtsp://192.168.2.118:554/stream1")
        self.assertEqual(transport.requests[1][0], "http://cam/onvif/device_service")

    def test_a_wrong_password_fails_immediately_without_probing_further(self) -> None:
        transport = _Transport(
            urllib.error.HTTPError("u", 401, "Unauthorized", {}, io.BytesIO(b"NotAuthorized"))
        )
        with mock.patch("scoop_ai.dashboard.discovery.urllib.request.urlopen", transport):
            with self.assertRaises(OnvifAuthError) as caught:
                fetch_stream_uri("http://cam/onvif/device_service", username="a", password="b")
        self.assertIn("rejected this username and password", str(caught.exception))
        # One rejected call is enough; retrying the same credentials is pointless.
        self.assertEqual(len(transport.requests), 1)

    def test_a_soap_fault_is_surfaced(self) -> None:
        fault = (
            '<?xml version="1.0"?>'
            '<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope"><s:Body>'
            "<s:Fault><s:Reason><s:Text>Action not supported</s:Text></s:Reason></s:Fault>"
            "</s:Body></s:Envelope>"
        ).encode()
        transport = _Transport(CAPABILITIES, fault)
        with mock.patch("scoop_ai.dashboard.discovery.urllib.request.urlopen", transport):
            with self.assertRaises(OnvifError) as caught:
                fetch_stream_uri("http://cam/onvif/device_service", username="a", password="b")
        self.assertIn("Action not supported", str(caught.exception))

    def test_a_camera_without_profiles_is_reported(self) -> None:
        empty = _envelope(f'<GetProfilesResponse xmlns="{MEDIA_NAMESPACE}"/>')
        transport = _Transport(CAPABILITIES, empty)
        with mock.patch("scoop_ai.dashboard.discovery.urllib.request.urlopen", transport):
            with self.assertRaises(OnvifError) as caught:
                fetch_stream_uri("http://cam/onvif/device_service", username="a", password="b")
        self.assertIn("did not report any video profiles", str(caught.exception))

    def test_an_unreachable_camera_explains_the_network_problem(self) -> None:
        # Capability probing falls back, so the media call reports the outage.
        transport = _Transport(
            urllib.error.URLError("timed out"), urllib.error.URLError("timed out")
        )
        with mock.patch("scoop_ai.dashboard.discovery.urllib.request.urlopen", transport):
            with self.assertRaises(OnvifError) as caught:
                fetch_stream_uri("http://cam/onvif/device_service", username="a", password="b")
        self.assertIn("could not be reached", str(caught.exception))

    def test_a_non_http_service_address_is_refused(self) -> None:
        with self.assertRaises(OnvifError):
            fetch_stream_uri("ftp://cam/service", username="a", password="b")


class WithCredentialsTests(unittest.TestCase):
    def test_embeds_credentials_and_keeps_the_port_and_path(self) -> None:
        self.assertEqual(
            with_credentials("rtsp://192.168.2.118:554/stream1", "admin", "pass"),
            "rtsp://admin:pass@192.168.2.118:554/stream1",
        )

    def test_escapes_characters_that_would_break_the_authority(self) -> None:
        self.assertEqual(
            with_credentials("rtsp://cam/stream", "user@site", "p@ss:word/1"),
            "rtsp://user%40site:p%40ss%3Aword%2F1@cam/stream",
        )

    def test_preserves_a_query_string(self) -> None:
        self.assertEqual(
            with_credentials("rtsp://cam:554/cam/realmonitor?channel=1&subtype=0", "a", "b"),
            "rtsp://a:b@cam:554/cam/realmonitor?channel=1&subtype=0",
        )

    def test_brackets_an_ipv6_host(self) -> None:
        self.assertEqual(
            with_credentials("rtsp://[fe80::1]:554/stream1", "a", "b"),
            "rtsp://a:b@[fe80::1]:554/stream1",
        )

    def test_a_missing_path_becomes_root(self) -> None:
        self.assertEqual(with_credentials("rtsp://cam:554", "a", "b"), "rtsp://a:b@cam:554/")


if __name__ == "__main__":
    unittest.main()
