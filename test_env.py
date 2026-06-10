from dotenv import load_dotenv
import os

load_dotenv()

print("PUBLIC:", os.getenv("LANGFUSE_PUBLIC_KEY"))
print("SECRET:", os.getenv("LANGFUSE_SECRET_KEY"))
print("BASE_URL:", os.getenv("LANGFUSE_BASE_URL"))