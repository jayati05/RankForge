"""Start the local single-worker service with RANKFORGE_* host/port settings."""

import uvicorn
from app.core.config import Settings


def main() -> None:
    settings = Settings.from_env()
    uvicorn.run(
        "app.api.main:app",
        host=settings.host,
        port=settings.port,
        workers=1,
        access_log=False,
        log_level=settings.log_level,
    )


if __name__ == "__main__":
    main()
