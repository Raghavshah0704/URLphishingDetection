import pandas as pd
import random

# -----------------------------
# BASE WORDS
# -----------------------------
brands = ["aadhaar", "uidai"]
actions = ["update", "verify", "login", "secure", "kyc", "check", "validate"]
extras = ["account", "card", "portal", "service"]
domains = [".com", ".net", ".org", ".xyz", ".ml", ".ga", ".cf"]

# -----------------------------
# GENERATE PHISHING URLs
# -----------------------------
phishing_urls = []

for _ in range(500):
    url = "http://"
    url += random.choice(brands) + "-"
    url += random.choice(actions) + "-"
    url += random.choice(extras)
    url += random.choice(domains)
    
    phishing_urls.append((url, 1))

# -----------------------------
# LEGIT URLs
# -----------------------------
legit_urls = [
    ("https://uidai.gov.in", 0),
    ("https://myaadhaar.uidai.gov.in", 0),
    ("https://resident.uidai.gov.in", 0),
]

# duplicate legit to balance
legit_urls = legit_urls * 50  # ~150 legit samples

# -----------------------------
# COMBINE
# -----------------------------
data = legit_urls + phishing_urls

df = pd.DataFrame(data, columns=["url", "status"])

# -----------------------------
# SAVE
# -----------------------------
df.to_csv("aadhar_dataset.csv", index=False)

print("Dataset created:", df.shape)