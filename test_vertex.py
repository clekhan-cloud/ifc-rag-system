from google import genai

client = genai.Client(
    vertexai=True,
    project="gd-gcp-gridu-genai",
    location="us-central1"
)

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Hello"
)

print(response.text)