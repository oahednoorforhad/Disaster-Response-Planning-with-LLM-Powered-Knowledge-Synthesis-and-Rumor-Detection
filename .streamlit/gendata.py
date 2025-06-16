# Generate 100 synthetic flood-related Twitter-style posts in JSONL format

import json
from datetime import datetime, timedelta, timezone
import random

# Sample Bangladeshi flood-affected areas
areas = [
    "Sunamganj", "Sylhet", "Gaibandha", "Rangpur", "Kurigram", "Jamalpur", "Bogura",
    "Netrokona", "Mymensingh", "Sirajganj", "Lalmonirhat", "Nilphamari", "Tangail",
    "Shariatpur", "Faridpur", "Barisal", "Patuakhali", "Bhola", "Cox's Bazar",
    "Khulna", "Satkhira", "Chandpur", "Habiganj", "Brahmanbaria", "Noakhali"
]

# Sample needs and hashtags
needs = [
    "need immediate rescue", "are stranded", "are out of food", "require medical assistance",
    "need shelter", "require urgent relief", "are submerged", "report no clean drinking water"
]
hashtags = ["#Flood2025", "#BangladeshFlood", "#RescueNeeded", "#DisasterRelief", "#HelpNow"]

# Generate 100 posts
posts = []
base_time = datetime.now(timezone.utc)

for i in range(100):
    area = random.choice(areas)
    need = random.choice(needs)
    hashtag_sample = random.sample(hashtags, k=2)
    text = f"Flood update: People in {area} {need}. Situation worsening. {' '.join(hashtag_sample)}"
    post = {
        "source": "Twitter",
        "author": f"user_{random.randint(1000,9999)}",
        "text": text,
        "timestamp": (base_time - timedelta(minutes=random.randint(0, 1440))).isoformat() + "Z",
        "location": area
    }
    posts.append(post)

# Save to JSONL
flood_jsonl_path = "/bangladesh_flood_posts_2025.jsonl"
with open(flood_jsonl_path, "w", encoding="utf-8") as f:
    for post in posts:
        f.write(json.dumps(post) + "\n")

flood_jsonl_path = flood_jsonl_path.replace("\\", "/")  # Ensure path is in correct format
print(f"Generated {len(posts)} synthetic flood-related posts in JSONL format at: {flood_jsonl_path}")
