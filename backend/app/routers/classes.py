from fastapi import APIRouter

router = APIRouter(prefix="/classes", tags=["classes"])

LAND_COVER_CLASSES = [
    {"id": 1, "name": "Forest",     "color": "#2d6a4f"},
    {"id": 2, "name": "Grassland",  "color": "#95d5b2"},
    {"id": 3, "name": "Cropland",   "color": "#e9c46a"},
    {"id": 4, "name": "Urban",      "color": "#adb5bd"},
    {"id": 5, "name": "Water",      "color": "#4895ef"},
    {"id": 6, "name": "Bare Soil",  "color": "#bc6c25"},
]


@router.get("", response_model=list[dict])
async def get_classes():
    """Return the list of land cover classes."""
    return LAND_COVER_CLASSES
