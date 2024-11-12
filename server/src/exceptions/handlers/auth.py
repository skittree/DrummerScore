from fastapi import Request, status
from fastapi.responses import RedirectResponse

from exceptions.auth import LoginRedirectException


async def login_redirect_handler(
        request: Request, exc: LoginRedirectException
) -> RedirectResponse:
    """
    При LoginRedirectException, перенаправляет
    пользователя на страницу логина.
    """
    url = request.url_for("auth_login_view")
    response = RedirectResponse(
        url, status_code=status.HTTP_302_FOUND)

    # if htmx request - let htmx handle redirect
    if request.headers.get("hx-request"):
        response.headers["HX-Redirect"] = str(url)
        response.status_code = status.HTTP_200_OK
    return response
