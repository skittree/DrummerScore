from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    HOST: str = "localhost"
    PORT: int = 8200
    BASE_DIR: Path = Path(__file__).parent.parent

    ACCESS_EXPIRES_MINUTES: int = 5
    ALLOWED_ORIGINS: list = ["https://localhost:8200"]


settings = Settings()
