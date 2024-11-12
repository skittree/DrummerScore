from fastapi import APIRouter

from views import home


view_router = APIRouter(tags=["Views"])
view_router.include_router(home.router)
