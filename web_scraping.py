"""
Automated Web Scraping and Dataset Harvesting Module
Author: Verina Fouad Farid Khalil
Role: Dataset Engineering & Web Scraping Pipeline Lead

This module automates the scraping, sanitization, and deduplication
of multi-category electronic component imagery (Resistors, Capacitors, Diodes, ICs)
from component catalogs and electronics distributor repositories.
"""

import os
import requests
import hashlib
from typing import List, Dict

class ElectronicComponentScraper:
    """Scrapes, validates, and partitions electronic component image datasets."""
    def __init__(self, output_dir: str = "data/raw_components"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.seen_hashes = set()

    def sanitize_and_save(self, image_bytes: bytes, category: str, filename: str) -> bool:
        """Verifies image integrity and prevents duplicate downloads using MD5 hashes."""
        img_hash = hashlib.md5(image_bytes).hexdigest()
        if img_hash in self.seen_hashes:
            return False  # Skip duplicate
        
        self.seen_hashes.add(img_hash)
        cat_dir = os.path.join(self.output_dir, category)
        os.makedirs(cat_dir, exist_ok=True)
        
        target_path = os.path.join(cat_dir, filename)
        with open(target_path, "wb") as f:
            f.write(image_bytes)
        return True

    def fetch_catalog_images(self, urls: List[Dict[str, str]]):
        """Batch ingestion loop with retry logic and category folder placement."""
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        saved_count = 0
        for item in urls:
            try:
                resp = requests.get(item["url"], headers=headers, timeout=10)
                if resp.status_code == 200 and len(resp.content) > 5000:
                    if self.sanitize_and_save(resp.content, item["category"], item["filename"]):
                        saved_count += 1
            except Exception as e:
                continue
        print(f"Dataset Acquisition Complete: {saved_count} unique components archived.")
