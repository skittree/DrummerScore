import gettext
from functools import lru_cache
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from services.templates import templates
from config import settings


@lru_cache(maxsize=15)
def load_translations(language_code):
    try:
        gnu_translations = gettext.translation(
            domain="messages",
            localedir=settings.BASE_DIR / "data" / "locale",
            languages=[language_code],
        )
    except FileNotFoundError:
        gnu_translations = gettext.translation(
            domain="messages",
            localedir=settings.BASE_DIR / "data" / "locale",
            languages=["en_US"],
            fallback=True,
        )
    return gnu_translations


class I18nMiddleware(BaseHTTPMiddleware):
    async def dispatch(
            self, request: Request, call_next):
        accept_language = request.headers.get("Accept-Language", "en_US")
        language_code = (
            request.cookies.get("locale")
            or "-".join(accept_language.split(",")[0].split("_")))
        gnu_translations = load_translations(language_code)
        templates.env.install_gettext_translations(gnu_translations)
        response = await call_next(request)
        return response
