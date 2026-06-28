"""
Scrape data center images using Brightdata Web Unlocker.

Targets image search results for DC components:
racks, servers, ports, cables, LEDs, serial number labels.

Usage:
    python scripts/scrape_dc_images.py
"""

import hashlib
import os
import re
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

API_TOKEN = os.getenv("BRIGHTDATA_API_TOKEN")
ZONE = os.getenv("BRIGHTDATA_ZONE", "web_unlocker1")
OUTPUT_DIR = Path("data/sample_images")

SEARCH_QUERIES = [
    "data center server rack front view",
    "data center server rack LED status lights",
    "server rack cable management close up",
    "data center server serial number label",
    "server rack ethernet ports cables",
    "data center rack mounted servers",
    "server rack front panel LEDs blinking",
    "data center cable patching panel",
    "server rack asset tag label",
    "network switch ports data center",
    "server rack rear view cables",
    "data center hot aisle cold aisle",
    "server chassis front bezel LEDs",
    "rack PDU power distribution unit",
    "fiber optic patch panel data center",
]


def scrape_url(url: str) -> str:
    response = requests.post(
        "https://api.brightdata.com/request",
        headers={
            "Authorization": f"Bearer {API_TOKEN}",
            "Content-Type": "application/json",
        },
        json={"zone": ZONE, "url": url, "format": "raw"},
        timeout=30,
    )
    response.raise_for_status()
    return response.text


def extract_image_urls(html: str) -> list[str]:
    patterns = [
        r'https?://[^"\s<>]+\.(?:jpg|jpeg|png|webp)',
        r'"ou":"(https?://[^"]+)"',
        r'data-src="(https?://[^"]+\.(?:jpg|jpeg|png|webp))"',
        r'"(https?://[^"]*(?:images|photos|img)[^"]*\.(?:jpg|jpeg|png|webp))"',
    ]
    urls = set()
    for pattern in patterns:
        for match in re.findall(pattern, html, re.IGNORECASE):
            url = match if isinstance(match, str) else match[0]
            if len(url) > 30 and "icon" not in url.lower() and "logo" not in url.lower():
                urls.add(url)
    return list(urls)


def download_image(url: str, output_dir: Path, prefix: str) -> bool:
    try:
        resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        content_type = resp.headers.get("content-type", "")
        if "image" not in content_type and not url.lower().endswith(
            (".jpg", ".jpeg", ".png", ".webp")
        ):
            return False
        if len(resp.content) < 10_000:
            return False

        ext = ".jpg"
        if "png" in content_type or url.lower().endswith(".png"):
            ext = ".png"
        elif "webp" in content_type or url.lower().endswith(".webp"):
            ext = ".webp"

        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        filename = f"{prefix}_{url_hash}{ext}"
        filepath = output_dir / filename
        filepath.write_bytes(resp.content)
        return True
    except Exception:
        return False


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    total_downloaded = 0

    for i, query in enumerate(SEARCH_QUERIES):
        prefix = f"dc_{i:02d}"
        search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}&tbm=isch&num=30"

        print(f"\n[{i+1}/{len(SEARCH_QUERIES)}] Searching: {query}")

        try:
            html = scrape_url(search_url)
            image_urls = extract_image_urls(html)
            print(f"  Found {len(image_urls)} image URLs")

            downloaded = 0
            for img_url in image_urls[:20]:
                if download_image(img_url, OUTPUT_DIR, prefix):
                    downloaded += 1

            total_downloaded += downloaded
            print(f"  Downloaded {downloaded} images")

        except Exception as e:
            print(f"  Error: {e}")

        time.sleep(1)

    print(f"\nTotal: {total_downloaded} images saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
