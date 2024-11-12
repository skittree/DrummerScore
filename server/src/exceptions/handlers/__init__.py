from fastapi import HTTPException, Request, Response

from services.templates import templates


async def custom_404_handler(
        request: Request, exc: HTTPException
) -> Response:
    """
    Перенаправляет на страницу 404 при ошибке.
    """
    return templates.TemplateResponse("core/404.html", {"request": request})
