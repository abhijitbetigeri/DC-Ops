"""Scrape NVIDIA GB200 NVL72 specific rack images."""

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

QUERIES = [
    "NVIDIA GB200 NVL72 server rack",
    "NVIDIA DGX GB200 rack front view",
    "NVIDIA NVL72 compute tray close up",
    "NVIDIA Blackwell GPU tray server",
    "NVIDIA NVLink switch tray rack",
    "NVIDIA GB200 NVL72 rear panel cables",
    "NVIDIA DGX server rack LED status",
    "NVIDIA GB200 liquid cooling manifold",
    "NVIDIA DGX SuperPOD rack",
    "NVIDIA ConnectX-7 OSFP port server",
    "NVIDIA GB300 NVL72 server rack",
    "data center GPU server rack front bezel",
    "server rack NVMe drive bay close up",
    "data center rack power shelf PSU",
    "server rack BMC management port RJ45",
]


def scrape_url(url: str) -> str:
    response = requests.post(
        "https://api.brightdata.com/request",
        headers={
            "Authorization": f"Bearer {API_TOKEN}",
            "Content-Type": "application/json",
        },
        json={"zone": ZONE, "url": url, "format": "raw"},
        timeout=60,
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

    for i, query in enumerate(QUERIES):
        prefix = f"nvl_{i:02d}"
        search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}&tbm=isch&num=30"

        print(f"\n[{i+1}/{len(QUERIES)}] Searching: {query}")

        try:
            html = scrape_url(search_url)
            image_urls = extract_image_urls(html)
            print(f"  Found {len(image_urls)} image URLs")

            downloaded = 0
            for img_url in image_urls[:15]:
                if download_image(img_url, OUTPUT_DIR, prefix):
                    downloaded += 1

            total_downloaded += downloaded
            print(f"  Downloaded {downloaded} images")

        except Exception as e:
            print(f"  Error: {e}")

        time.sleep(2)

    existing = len(list(OUTPUT_DIR.glob("*")))
    print(f"\nNVL72 scrape: {total_downloaded} new images")
    print(f"Total in {OUTPUT_DIR}/: {existing} images")


if __name__ == "__main__":
    main()
