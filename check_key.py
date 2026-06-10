# check_key.py

import os

key = os.getenv("Gemini API Key")

print("KEY FOUND:", bool(key))

if key:
    print("PREFIX:", key[:10])