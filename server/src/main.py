from pathlib import Path

import uvicorn
from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from exceptions.auth import LoginRedirectException
from exceptions.handlers import custom_404_handler
from exceptions.handlers.auth import login_redirect_handler
from middleware.i18n import I18nMiddleware

from config import settings
from api import api_router
from views import view_router


app = FastAPI()

# static files
app.mount(
    "/static",
    StaticFiles(directory=Path(__file__).parent.parent / "data" / "static"),
    name="static",
)


# routers
app.include_router(api_router)
app.include_router(view_router)

# middlewares
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(I18nMiddleware)

# exception handlers
app.add_exception_handler(status.HTTP_404_NOT_FOUND, custom_404_handler)
app.add_exception_handler(
    status.HTTP_405_METHOD_NOT_ALLOWED, custom_404_handler
)
app.add_exception_handler(LoginRedirectException, login_redirect_handler)


if __name__ == "__main__":
    uvicorn.run(
        "main:app", host=settings.HOST, port=settings.PORT, reload=True
    )
