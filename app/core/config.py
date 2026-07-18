import os
from dotenv import load_dotenv

load_dotenv()  # loads variables from .env file into environment


class Settings:
    """
    Centralized application settings.
    Values are pulled from environment variables (.env locally, or real
    environment variables when deployed to Render/production).
    """

    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")

    # Comma-separated list of allowed frontend origins, e.g.:
    # "http://localhost:5173,https://your-frontend.vercel.app"
    CORS_ORIGINS: list[str] = os.getenv(
        "CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173"
    ).split(",")


settings = Settings()