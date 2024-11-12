import csv
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from services.templates import templates

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def home(request: Request) -> HTMLResponse:
    with open(Path(__file__).parent / "onsets.csv", newline="") as csvfile:
        spamreader = csv.reader(csvfile, delimiter=",")
        spamreader.__next__()
        onsets = []
        for row in spamreader:
            onsets.append(
                {
                    "onset": float(row[0]),
                    "bass": row[1] == "True",
                    "snare": row[2] == "True",
                    "hihat": row[3] == "True",
                    "tom": row[4] == "True",
                    "crash": row[5] == "True",
                    "ride": row[6] == "True",
                }
            )
    return templates.TemplateResponse(
        request, "home.html", {"onsets": enumerate(onsets)}
    )
