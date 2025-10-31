import os
from dotenv import load_dotenv

print("=== Environment Variable Debug ===")
print(f"Current working directory: {os.getcwd()}")

# Check if .env file exists
env_file = '.env'
print(f".env file exists: {os.path.exists(env_file)}")

if os.path.exists(env_file):
    with open(env_file, 'r') as f:
        content = f.read()
        print(f".env file content:\n{content}")

# Load environment variables
load_dotenv()

# Check what was loaded
api_key = os.getenv('OPENAI_API_KEY')
print(f"API Key loaded: {'Yes' if api_key else 'No'}")
if api_key:
    print(f"API Key starts with: {api_key[:15]}...")
    print(f"API Key length: {len(api_key)}")
else:
    print("No API key found")

# Check all environment variables with OPENAI
openai_vars = {k: v for k, v in os.environ.items() if 'OPENAI' in k.upper()}
print(f"OpenAI related env vars: {openai_vars}")
