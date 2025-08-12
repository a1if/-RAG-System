import os
from dotenv import load_dotenv

REQUIRED_ENV_VARS = ["GEMINI_API_KEY"]

def validate_env():
    load_dotenv()
    missing = [var for var in REQUIRED_ENV_VARS if not os.getenv(var)]
    if missing:
        raise EnvironmentError(f"❌ Missing required environment variables: {', '.join(missing)}")