# -*- coding: utf-8 -*-
"""
URL Feature Extractor (Clean Dataset Compatible)
=================================================
Takes a raw URL → fetches the page → returns a dict of the 19 features
that match the feature_columns.json used by the trained models.

Features: domain_length, has_hyphen, url_shortener, suspicious_words,
domain_age_category, dns_record, iframe, popup_window, onmouseover,
num_forms, external_scripts, external_images, hidden_elements,
internal_links, external_links, null_links, external_ratio,
script_length, domain_entropy
"""

import re
import math
import socket
import warnings
from urllib.parse import urlparse, urljoin

import numpy as np
import requests
import tldextract
from bs4 import BeautifulSoup
import urllib3

warnings.filterwarnings("ignore")
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ─────────────────────────────────────────────────────────────────────────────
# HTTP SESSION
# ─────────────────────────────────────────────────────────────────────────────
_session = requests.Session()
_session.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
})

# Known URL shortener domains
URL_SHORTENERS = {
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd",
    "buff.ly", "rebrand.ly", "cutt.ly", "short.io", "tiny.cc",
    "bl.ink", "lnkd.in", "youtu.be", "shorturl.at",
}

# Suspicious words commonly found in phishing URLs
SUSPICIOUS_WORDS = [
    "login", "signin", "verify", "account", "update", "secure",
    "banking", "confirm", "password", "suspend", "alert", "unusual",
    "restrict", "unlock", "validate", "authenticate", "credential",
    "recover", "restore", "expire", "reactivate", "urgent", "immediately",
    "paypal", "apple", "microsoft", "google", "amazon", "netflix",
    "facebook", "instagram", "whatsapp", "dropbox",
]


def _normalize_url(url: str) -> str:
    url = str(url).strip()
    if not url.startswith(("http://", "https://")):
        url = "http://" + url
    return url


def _domain_entropy(domain: str) -> float:
    """Shannon entropy of domain string characters."""
    if not domain:
        return 0.0
    freq = {}
    for c in domain.lower():
        freq[c] = freq.get(c, 0) + 1
    length = len(domain)
    entropy = -sum((count / length) * math.log2(count / length) for count in freq.values())
    return round(entropy, 4)


def _check_dns(domain: str) -> int:
    """Check if domain has DNS record. Returns 1 if resolves, 0 otherwise."""
    try:
        socket.getaddrinfo(domain, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
        return 1
    except (socket.gaierror, socket.herror, OSError):
        return 0


def _get_domain_age_category(domain: str) -> int:
    """
    Domain age category (approximation without WHOIS):
    0 = unknown/new, 1 = medium, 2 = established
    Uses DNS resolution as a rough proxy.
    """
    # Without a WHOIS lookup, we approximate:
    # If DNS resolves → likely established (1)
    # Otherwise → unknown (0)
    return 1 if _check_dns(domain) else 0


def extract_features_from_url(url: str, timeout: int = 10) -> dict:
    """
    Extract the 19 features from a raw URL matching the trained model's
    feature_columns.json.

    Returns a dict with keys:
        domain_length, has_hyphen, url_shortener, suspicious_words,
        domain_age_category, dns_record, iframe, popup_window, onmouseover,
        num_forms, external_scripts, external_images, hidden_elements,
        internal_links, external_links, null_links, external_ratio,
        script_length, domain_entropy
    """
    url = _normalize_url(url)
    parsed = urlparse(url)
    ext = tldextract.extract(url)
    domain = f"{ext.domain}.{ext.suffix}" if ext.suffix else ext.domain
    full_domain = parsed.netloc or domain

    feat = {}

    # =========================================================================
    # URL-BASED FEATURES
    # =========================================================================

    # domain_length: length of the registered domain (e.g., "google.com" = 10)
    feat["domain_length"] = len(domain)

    # has_hyphen: 1 if domain contains a hyphen
    feat["has_hyphen"] = 1 if "-" in (ext.domain or "") else 0

    # url_shortener: 1 if the domain is a known URL shortener
    feat["url_shortener"] = 1 if domain.lower() in URL_SHORTENERS else 0

    # suspicious_words: count of suspicious/phishing keywords in the URL
    url_lower = url.lower()
    feat["suspicious_words"] = sum(1 for w in SUSPICIOUS_WORDS if w in url_lower)

    # domain_age_category: approximate age category (0=new/unknown, 1=established)
    feat["domain_age_category"] = _get_domain_age_category(full_domain)

    # dns_record: 1 if the domain resolves via DNS
    feat["dns_record"] = _check_dns(full_domain)

    # domain_entropy: Shannon entropy of the domain characters
    feat["domain_entropy"] = _domain_entropy(ext.domain or "")

    # =========================================================================
    # HTML / CONTENT FEATURES — fetch the page
    # =========================================================================
    html = ""
    try:
        resp = _session.get(url, timeout=timeout, verify=False, allow_redirects=True)
        ct = resp.headers.get("Content-Type", "")
        if "charset" not in ct.lower():
            resp.encoding = resp.apparent_encoding
        html = resp.text or ""
    except Exception:
        pass

    if not html:
        # Default all HTML features to 0
        feat["iframe"] = 0
        feat["popup_window"] = 0
        feat["onmouseover"] = 0
        feat["num_forms"] = 0
        feat["external_scripts"] = 0
        feat["external_images"] = 0
        feat["hidden_elements"] = 0
        feat["internal_links"] = 0
        feat["external_links"] = 0
        feat["null_links"] = 0
        feat["external_ratio"] = 0.0
        feat["script_length"] = 0
        return feat

    try:
        soup = BeautifulSoup(html, "html.parser")

        # iframe: number of iframes
        feat["iframe"] = len(soup.find_all("iframe"))

        # popup_window: count of window.open calls
        feat["popup_window"] = len(re.findall(r'window\.(open|showModalDialog)\s*\(', html))

        # onmouseover: 1 if page uses onmouseover to manipulate status bar
        feat["onmouseover"] = 1 if re.search(r'onmouseover\s*=', html, re.I) else 0

        # num_forms: number of form elements
        feat["num_forms"] = len(soup.find_all("form"))

        # Count scripts — external vs inline
        scripts = soup.find_all("script")
        external_script_count = 0
        total_script_length = 0
        for s in scripts:
            if s.get("src"):
                external_script_count += 1
            if s.string:
                total_script_length += len(s.string)
        feat["external_scripts"] = external_script_count
        feat["script_length"] = total_script_length

        # External images
        images = soup.find_all("img")
        ext_img_count = 0
        for img in images:
            src = (img.get("src") or "").strip()
            if src:
                try:
                    abs_src = urljoin(url, src)
                    img_ext = tldextract.extract(abs_src)
                    img_root = f"{img_ext.domain}.{img_ext.suffix}"
                    if img_root and img_root != domain:
                        ext_img_count += 1
                except Exception:
                    pass
        feat["external_images"] = ext_img_count

        # Hidden elements
        hidden_inputs = sum(
            1 for inp in soup.find_all("input")
            if (inp.get("type", "") or "").lower() == "hidden"
        )
        hidden_divs = sum(
            1 for el in soup.find_all(style=True)
            if "display:none" in (el.get("style", "") or "").replace(" ", "").lower()
            or "visibility:hidden" in (el.get("style", "") or "").replace(" ", "").lower()
        )
        feat["hidden_elements"] = hidden_inputs + hidden_divs

        # Links: internal, external, null
        all_links = soup.find_all("a", href=True)
        internal = 0
        external = 0
        null_links = 0

        for a in all_links:
            href = (a.get("href") or "").strip()
            if not href or href in ("#", "javascript:void(0)", "javascript:", ""):
                null_links += 1
                continue
            try:
                abs_href = urljoin(url, href)
                link_ext = tldextract.extract(abs_href)
                link_root = f"{link_ext.domain}.{link_ext.suffix}"
                if link_root == domain:
                    internal += 1
                else:
                    external += 1
            except Exception:
                null_links += 1

        feat["internal_links"] = internal
        feat["external_links"] = external
        feat["null_links"] = null_links

        # external_ratio: ratio of external links to total links
        total_links = internal + external + null_links
        feat["external_ratio"] = round(external / max(total_links, 1), 4)

    except Exception:
        feat.setdefault("iframe", 0)
        feat.setdefault("popup_window", 0)
        feat.setdefault("onmouseover", 0)
        feat.setdefault("num_forms", 0)
        feat.setdefault("external_scripts", 0)
        feat.setdefault("external_images", 0)
        feat.setdefault("hidden_elements", 0)
        feat.setdefault("internal_links", 0)
        feat.setdefault("external_links", 0)
        feat.setdefault("null_links", 0)
        feat.setdefault("external_ratio", 0.0)
        feat.setdefault("script_length", 0)

    return feat


# ─────────────────────────────────────────────────────────────────────────────
# CLI TEST
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json
    import sys

    test_url = sys.argv[1] if len(sys.argv) > 1 else "https://www.google.com"
    print(f"\nExtracting features for: {test_url}\n")
    features = extract_features_from_url(test_url)
    print(json.dumps(features, indent=2))
    print(f"\nTotal features: {len(features)}")
