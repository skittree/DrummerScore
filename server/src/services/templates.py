import gettext
from fastapi.templating import Jinja2Templates

from config import settings

try:
    gnu_translations = gettext.translation(
        domain="messages",
        localedir=settings.BASE_DIR / "data" / "locale",
        languages=["en_US", "ru_RU"],
    )
except FileNotFoundError:
    raise RuntimeError("Please compile messages first.")


templates = Jinja2Templates(
    directory=settings.BASE_DIR / "data" / "templates",
    trim_blocks=True, lstrip_blocks=True,
    extensions=["jinja2.ext.i18n"])
templates.env.install_gettext_translations(gnu_translations)
