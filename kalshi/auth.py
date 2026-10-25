"""Kalshi RSA-PSS request signing, config-free.

Takes the key id and key path explicitly so a project with its own settings (~/Kalshi_BTC) can
sign without importing the equity `config`. `kalshi.client` delegates its signing here.
"""
import base64
import time

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

_KEYS = {}


def load_private_key(path):
    """PEM private key, loaded once per path."""
    key = _KEYS.get(path)
    if key is None:
        with open(path, "rb") as fh:
            key = serialization.load_pem_private_key(fh.read(), password=None)
        _KEYS[path] = key
    return key


def sign(private_key, timestamp_ms, method, path):
    """RSA-PSS-SHA256 over `{ts}{METHOD}{path}`. `path` includes /trade-api/v2, excludes query."""
    msg = f"{timestamp_ms}{method.upper()}{path}".encode()
    sig = private_key.sign(
        msg,
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=hashes.SHA256.digest_size),
        hashes.SHA256(),
    )
    return base64.b64encode(sig).decode()


def auth_headers(key_id, key_path, method, path):
    ts = str(int(time.time() * 1000))
    return {
        "KALSHI-ACCESS-KEY": key_id,
        "KALSHI-ACCESS-SIGNATURE": sign(load_private_key(key_path), ts, method, path),
        "KALSHI-ACCESS-TIMESTAMP": ts,
        "Content-Type": "application/json",
    }
