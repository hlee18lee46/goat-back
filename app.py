# app.py
# FastAPI wrapper for MBTA v3 JSON:API + simple “route_options” demo ranking
#
# Run:
#   python -m venv venv
#   source venv/bin/activate
#   pip install fastapi uvicorn httpx python-dotenv pydantic
#   export MBTA_API_KEY="YOUR_KEY"   # optional but recommended
#   uvicorn app:app --reload
#
# Example calls:
#   http://127.0.0.1:8000/mbta/routes?stop=place-sstat
#   http://127.0.0.1:8000/mbta/stops_search?q=south&limit=5
#   http://127.0.0.1:8000/mbta/shapes?route=Red
#   http://127.0.0.1:8000/route_options?origin=node-sstat-411stair-sl&dest=71630

from __future__ import annotations

import os
import time
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# -----------------------------
# App + CORS
# -----------------------------

app = FastAPI()  # MUST come first
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # allow all origins
    allow_credentials=True,
    allow_methods=["*"],      # allow all HTTP methods
    allow_headers=["*"],      # allow all headers
)

load_dotenv()

MBTA_BASE_URL = os.getenv("MBTA_BASE_URL", "https://api-v3.mbta.com")
MBTA_API_KEY = os.getenv("MBTA_API_KEY")  # optional, but MBTA recommends using a key
DEFAULT_TIMEOUT = float(os.getenv("HTTP_TIMEOUT_SECONDS", "15"))

# NOTE: OTP is not required for this demo ranking. Keep placeholder if you later add OTP.
OTP_BASE_URL = os.getenv("MBTA_OTP_BASE_URL", "https://api-v3.mbta.com/otp")  # placeholder


# -----------------------------
# JSON:API document models
# -----------------------------

class JsonApiLinks(BaseModel):
    self: Optional[str] = None
    prev: Optional[str] = None
    next: Optional[str] = None
    first: Optional[str] = None
    last: Optional[str] = None


class JsonApiResourceIdentifier(BaseModel):
    type: str
    id: str


class JsonApiRelationshipLinks(BaseModel):
    self: Optional[str] = None
    related: Optional[str] = None


class JsonApiRelationship(BaseModel):
    links: Optional[JsonApiRelationshipLinks] = None
    data: Optional[Union[JsonApiResourceIdentifier, List[JsonApiResourceIdentifier]]] = None


class FacilityProperty(BaseModel):
    name: str
    value: Union[str, int, float, None] = None


class LiveFacilityAttributes(BaseModel):
    updated_at: Optional[str] = None
    properties: List[FacilityProperty] = Field(default_factory=list)


class LiveFacilityResource(BaseModel):
    type: Literal["live_facility"] = "live_facility"
    id: str
    attributes: Optional[LiveFacilityAttributes] = None
    relationships: Dict[str, JsonApiRelationship] = Field(default_factory=dict)
    links: Dict[str, Any] = Field(default_factory=dict)


class TripAttributes(BaseModel):
    wheelchair_accessible: Optional[int] = None
    revenue_status: Optional[str] = None
    name: Optional[str] = None
    headsign: Optional[str] = None
    direction_id: Optional[int] = None
    block_id: Optional[str] = None
    bikes_allowed: Optional[int] = None


class TripResource(BaseModel):
    type: Literal["trip"] = "trip"
    id: str
    attributes: Optional[TripAttributes] = None
    relationships: Dict[str, JsonApiRelationship] = Field(default_factory=dict)
    links: Dict[str, Any] = Field(default_factory=dict)


class VehicleCarriage(BaseModel):
    occupancy_status: Optional[str] = None
    occupancy_percentage: Optional[int] = None
    label: Optional[str] = None


class VehicleAttributes(BaseModel):
    updated_at: Optional[str] = None
    speed: Optional[float] = None
    revenue_status: Optional[str] = None
    occupancy_status: Optional[str] = None
    longitude: Optional[float] = None
    latitude: Optional[float] = None
    label: Optional[str] = None
    direction_id: Optional[int] = None
    current_stop_sequence: Optional[int] = None
    current_status: Optional[str] = None
    bearing: Optional[int] = None
    carriages: Optional[List[VehicleCarriage]] = None


class VehicleResource(BaseModel):
    type: Literal["vehicle"] = "vehicle"
    id: str
    attributes: Optional[VehicleAttributes] = None
    relationships: Dict[str, JsonApiRelationship] = Field(default_factory=dict)
    links: Dict[str, Any] = Field(default_factory=dict)


class ServiceAttributes(BaseModel):
    valid_days: Optional[List[int]] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    schedule_type: Optional[str] = None
    schedule_name: Optional[str] = None
    description: Optional[str] = None
    schedule_typicality: Optional[int] = None
    added_dates: Optional[List[str]] = None
    removed_dates: Optional[List[str]] = None
    rating_start_date: Optional[str] = None
    rating_end_date: Optional[str] = None
    rating_description: Optional[str] = None


class ServiceResource(BaseModel):
    type: Literal["service"] = "service"
    id: str
    attributes: Optional[ServiceAttributes] = None
    relationships: Dict[str, JsonApiRelationship] = Field(default_factory=dict)
    links: Dict[str, Any] = Field(default_factory=dict)


class RoutePatternAttributes(BaseModel):
    typicality: Optional[int] = None
    time_desc: Optional[str] = None
    sort_order: Optional[int] = None
    name: Optional[str] = None
    direction_id: Optional[int] = None
    canonical: Optional[bool] = None


class RoutePatternResource(BaseModel):
    type: Literal["route_pattern"] = "route_pattern"
    id: str
    attributes: Optional[RoutePatternAttributes] = None
    relationships: Dict[str, JsonApiRelationship] = Field(default_factory=dict)
    links: Dict[str, Any] = Field(default_factory=dict)


class AlertActivePeriod(BaseModel):
    start: Optional[str] = None
    end: Optional[str] = None


class InformedEntity(BaseModel):
    trip: Optional[str] = None
    stop: Optional[str] = None
    route_type: Optional[int] = None
    route: Optional[str] = None
    facility: Optional[str] = None
    direction_id: Optional[int] = None
    activities: Optional[List[str]] = None


class AlertAttributes(BaseModel):
    url: Optional[str] = None
    updated_at: Optional[str] = None
    timeframe: Optional[str] = None
    short_header: Optional[str] = None
    severity: Optional[int] = None
    service_effect: Optional[str] = None
    reminder_times: Optional[List[str]] = None
    lifecycle: Optional[str] = None
    last_push_notification_timestamp: Optional[str] = None
    informed_entity: Optional[List[InformedEntity]] = None
    image_alternative_text: Optional[str] = None
    image: Optional[str] = None
    header: Optional[str] = None
    effect_name: Optional[str] = None
    effect: Optional[str] = None
    duration_certainty: Optional[str] = None
    description: Optional[str] = None
    created_at: Optional[str] = None
    closed_timestamp: Optional[str] = None
    cause: Optional[str] = None
    banner: Optional[str] = None
    active_period: Optional[List[AlertActivePeriod]] = None


class AlertResource(BaseModel):
    type: Literal["alert"] = "alert"
    id: str
    attributes: Optional[AlertAttributes] = None
    relationships: Dict[str, JsonApiRelationship] = Field(default_factory=dict)
    links: Dict[str, Any] = Field(default_factory=dict)


class StopAttributes(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    address: Optional[str] = None
    municipality: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    location_type: Optional[int] = None
    vehicle_type: Optional[int] = None
    wheelchair_boarding: Optional[int] = None
    platform_name: Optional[str] = None
    platform_code: Optional[str] = None
    on_street: Optional[str] = None
    at_street: Optional[str] = None


class StopResource(BaseModel):
    type: Literal["stop"] = "stop"
    id: str
    attributes: Optional[StopAttributes] = None
    relationships: Dict[str, JsonApiRelationship] = Field(default_factory=dict)
    links: Dict[str, Any] = Field(default_factory=dict)


class FacilityAttributes(BaseModel):
    type: Optional[str] = None
    short_name: Optional[str] = None
    long_name: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    properties: List[FacilityProperty] = Field(default_factory=list)


class FacilityResource(BaseModel):
    type: Literal["facility"] = "facility"
    id: str
    attributes: Optional[FacilityAttributes] = None
    relationships: Dict[str, JsonApiRelationship] = Field(default_factory=dict)
    links: Dict[str, Any] = Field(default_factory=dict)


class PredictionAttributes(BaseModel):
    update_type: Optional[str] = None
    stop_sequence: Optional[int] = None
    status: Optional[str] = None
    schedule_relationship: Optional[str] = None
    revenue_status: Optional[str] = None
    direction_id: Optional[int] = None
    departure_uncertainty: Optional[int] = None
    departure_time: Optional[str] = None
    arrival_uncertainty: Optional[int] = None
    arrival_time: Optional[str] = None


class PredictionResource(BaseModel):
    type: Literal["prediction"] = "prediction"
    id: str
    attributes: Optional[PredictionAttributes] = None
    relationships: Dict[str, JsonApiRelationship] = Field(default_factory=dict)
    links: Dict[str, Any] = Field(default_factory=dict)


# -----------------------------
# HTTP client + helpers
# -----------------------------

def _headers() -> Dict[str, str]:
    h: Dict[str, str] = {"Accept": "application/vnd.api+json"}
    if MBTA_API_KEY:
        h["x-api-key"] = MBTA_API_KEY
    return h


async def mbta_get(path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    url = f"{MBTA_BASE_URL.rstrip('/')}/{path.lstrip('/')}"
    try:
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT, headers=_headers()) as client:
            resp = await client.get(url, params=params)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Network error calling MBTA: {e}") from e

    if resp.status_code >= 400:
        try:
            payload = resp.json()
        except Exception:
            payload = {"detail": resp.text}
        raise HTTPException(status_code=resp.status_code, detail=payload)

    try:
        return resp.json()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"MBTA returned non-JSON response: {e}") from e


def jsonapi_list(payload: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    data = payload.get("data")
    included = payload.get("included") or []
    links = payload.get("links") or {}
    if data is None:
        return [], included, links
    if isinstance(data, list):
        return data, included, links
    return [data], included, links


# -----------------------------
# Core endpoints
# -----------------------------

@app.get("/health")
def health():
    return {"ok": True}


# -------- Vehicles --------
@app.get("/mbta/vehicles")
async def get_vehicles(
    route: Optional[str] = Query(None, description="filter[route]=..."),
    stop: Optional[str] = Query(None, description="filter[stop]=..."),
    trip: Optional[str] = Query(None, description="filter[trip]=..."),
    direction_id: Optional[int] = Query(None, description="filter[direction_id]=0|1"),
    include: Optional[str] = Query("trip,stop,route", description="include=trip,stop,route"),
    page_limit: int = Query(50, ge=1, le=200),
):
    params: Dict[str, Any] = {"include": include, "page[limit]": page_limit}
    if route:
        params["filter[route]"] = route
    if stop:
        params["filter[stop]"] = stop
    if trip:
        params["filter[trip]"] = trip
    if direction_id is not None:
        params["filter[direction_id]"] = direction_id

    payload = await mbta_get("/vehicles", params=params)
    data_list, included, links = jsonapi_list(payload)

    vehicles = [VehicleResource.model_validate(d).model_dump() for d in data_list]
    return {"links": links, "data": vehicles, "included": included}


# -------- Predictions --------
@app.get("/mbta/predictions")
async def get_predictions(
    route: Optional[str] = Query(None, description="filter[route]=..."),
    stop: Optional[str] = Query(None, description="filter[stop]=..."),
    trip: Optional[str] = Query(None, description="filter[trip]=..."),
    direction_id: Optional[int] = Query(None, description="filter[direction_id]=0|1"),
    include: Optional[str] = Query("trip,stop,route,vehicle,alerts", description="include=..."),
    sort: Optional[str] = Query("departure_time", description="sort=departure_time or -departure_time"),
    page_limit: int = Query(50, ge=1, le=200),
):
    params: Dict[str, Any] = {"include": include, "sort": sort, "page[limit]": page_limit}
    if route:
        params["filter[route]"] = route
    if stop:
        params["filter[stop]"] = stop
    if trip:
        params["filter[trip]"] = trip
    if direction_id is not None:
        params["filter[direction_id]"] = direction_id

    payload = await mbta_get("/predictions", params=params)
    data_list, included, links = jsonapi_list(payload)

    preds = [PredictionResource.model_validate(d).model_dump() for d in data_list]
    return {"links": links, "data": preds, "included": included}


# -------- Trips (single) --------
@app.get("/mbta/trips/{trip_id}")
async def get_trip(
    trip_id: str,
    include: Optional[str] = Query("route,route_pattern,service,shape", description="include=..."),
):
    payload = await mbta_get(f"/trips/{trip_id}", params={"include": include})
    data_list, included, links = jsonapi_list(payload)
    if not data_list:
        raise HTTPException(status_code=404, detail="Trip not found")

    trip_obj = TripResource.model_validate(data_list[0]).model_dump()
    return {"links": links, "data": trip_obj, "included": included}


# -------- Stops (generic) --------
@app.get("/mbta/stops")
async def get_stops(
    route: Optional[str] = Query(None),
    location_type: Optional[int] = Query(None),
    latitude: Optional[float] = Query(None, alias="filter[latitude]"),
    longitude: Optional[float] = Query(None, alias="filter[longitude]"),
    radius: Optional[float] = Query(None, alias="filter[radius]"),
    sort: Optional[str] = Query(None),
    page_limit: int = Query(50, ge=1, le=200),
):
    params: Dict[str, Any] = {"page[limit]": page_limit}
    if route:
        params["filter[route]"] = route
    if location_type is not None:
        params["filter[location_type]"] = location_type
    if latitude is not None:
        params["filter[latitude]"] = latitude
    if longitude is not None:
        params["filter[longitude]"] = longitude
    if radius is not None:
        params["filter[radius]"] = radius
    if sort:
        params["sort"] = sort
    return await mbta_get("/stops", params=params)


# ✅ NEW: Routes wrapper (you were missing this)
@app.get("/mbta/routes")
async def get_routes(
    stop: Optional[str] = Query(None, description="filter[stop]=place-... or node-..."),
    route_type: Optional[int] = Query(None, description="filter[type]=0..4"),
    date: Optional[str] = Query(None, description="filter[date]=YYYY-MM-DD"),
    listed_route: Optional[bool] = Query(True, description="filter[listed_route]=true|false"),
    include: Optional[str] = Query(None, description="include=stop,line,route_patterns"),
    page_limit: int = Query(50, ge=1, le=200),
    sort: Optional[str] = Query(None),
):
    params: Dict[str, Any] = {"page[limit]": page_limit}

    if stop:
        params["filter[stop]"] = stop
    if route_type is not None:
        params["filter[type]"] = str(route_type)
    if date:
        params["filter[date]"] = date
    if listed_route is not None:
        params["filter[listed_route]"] = "true" if listed_route else "false"
    if include:
        params["include"] = include
    if sort:
        params["sort"] = sort

    return await mbta_get("/routes", params=params)


# -------- Facilities --------
@app.get("/mbta/facilities")
async def get_facilities(
    stop: Optional[str] = Query(None, description="filter[stop]=place-sstat, etc"),
    facility_type: Optional[str] = Query(None, description="filter[type]=ELEVATOR|ESCALATOR|..."),
    include: Optional[str] = Query("stop", description="include=stop"),
    page_limit: int = Query(50, ge=1, le=200),
):
    params: Dict[str, Any] = {"include": include, "page[limit]": page_limit}
    if stop:
        params["filter[stop]"] = stop
    if facility_type:
        params["filter[type]"] = facility_type

    payload = await mbta_get("/facilities", params=params)
    data_list, included, links = jsonapi_list(payload)

    facilities = [FacilityResource.model_validate(d).model_dump() for d in data_list]
    return {"links": links, "data": facilities, "included": included}


# -------- Live Facilities --------
@app.get("/mbta/live_facilities")
async def get_live_facilities(
    facility: Optional[str] = Query(None, description="filter[id]=<facility_id> (best-effort)"),
    page_limit: int = Query(50, ge=1, le=200),
):
    params: Dict[str, Any] = {"page[limit]": page_limit}
    if facility:
        params["filter[id]"] = facility

    payload = await mbta_get("/live_facilities", params=params)
    data_list, included, links = jsonapi_list(payload)

    live = [LiveFacilityResource.model_validate(d).model_dump() for d in data_list]
    return {"links": links, "data": live, "included": included}


# -------- Alerts --------
@app.get("/mbta/alerts")
async def get_alerts(
    route: Optional[str] = Query(None, description="filter[route]=..."),
    stop: Optional[str] = Query(None, description="filter[stop]=..."),
    facility: Optional[str] = Query(None, description="filter[facility]=..."),
    route_type: Optional[int] = Query(None, description="filter[route_type]=0..4"),
    include: Optional[str] = Query("facility", description="include=facility"),
    page_limit: int = Query(50, ge=1, le=200),
):
    params: Dict[str, Any] = {"include": include, "page[limit]": page_limit}
    if route:
        params["filter[route]"] = route
    if stop:
        params["filter[stop]"] = stop
    if facility:
        params["filter[facility]"] = facility
    if route_type is not None:
        params["filter[route_type]"] = route_type

    payload = await mbta_get("/alerts", params=params)
    data_list, included, links = jsonapi_list(payload)

    alerts = [AlertResource.model_validate(d).model_dump() for d in data_list]
    return {"links": links, "data": alerts, "included": included}


# -------- Route Patterns --------
@app.get("/mbta/route_patterns")
async def get_route_patterns(
    route: Optional[str] = Query(None, description="filter[route]=..."),
    direction_id: Optional[int] = Query(None, description="filter[direction_id]=0|1"),
    include: Optional[str] = Query("route,representative_trip", description="include=..."),
    page_limit: int = Query(50, ge=1, le=200),
):
    params: Dict[str, Any] = {"include": include, "page[limit]": page_limit}
    if route:
        params["filter[route]"] = route
    if direction_id is not None:
        params["filter[direction_id]"] = direction_id

    payload = await mbta_get("/route_patterns", params=params)
    data_list, included, links = jsonapi_list(payload)

    patterns = [RoutePatternResource.model_validate(d).model_dump() for d in data_list]
    return {"links": links, "data": patterns, "included": included}


# -----------------------------
# Stops search (local cache)
# -----------------------------

_STOPS_CACHE: Dict[str, Any] = {"ts": 0.0, "data": []}
_STOPS_CACHE_TTL_SECONDS = 60 * 60 * 6  # 6 hours


async def _get_all_stops_cached() -> List[Dict[str, Any]]:
    now = time.time()
    if _STOPS_CACHE["data"] and (now - _STOPS_CACHE["ts"] < _STOPS_CACHE_TTL_SECONDS):
        return _STOPS_CACHE["data"]

    all_stops: List[Dict[str, Any]] = []
    offset = 0
    limit = 1000

    while True:
        payload = await mbta_get("/stops", params={"page[limit]": limit, "page[offset]": offset})
        data = payload.get("data") or []
        if not data:
            break
        all_stops.extend(data)
        if len(data) < limit:
            break
        offset += limit

    _STOPS_CACHE["ts"] = now
    _STOPS_CACHE["data"] = all_stops
    return all_stops


@app.get("/mbta/stops_search")
async def stops_search(
    q: str = Query(..., min_length=2),
    limit: int = Query(5, ge=1, le=20),
):
    q_lower = q.strip().lower()
    stops = await _get_all_stops_cached()

    matches = [
        s for s in stops
        if (s.get("attributes", {}).get("name") or "").lower().find(q_lower) != -1
    ][:limit]

    return {"data": matches}


# -----------------------------
# Shapes (fix: accept both ?route= and ?filter[route]=)
# -----------------------------

@app.get("/mbta/shapes")
async def get_shapes(
    route: Optional[str] = Query(None, description="Route id (ex: Red, 1, SL1)"),
    filter_route: Optional[str] = Query(None, alias="filter[route]", description="Also supported"),
    page_limit: int = Query(1, ge=1, le=10),
    page_offset: int = Query(0, ge=0),
):
    rid = route or filter_route
    if not rid:
        raise HTTPException(status_code=422, detail="Missing required query param: route")

    params = {
        "filter[route]": rid,
        "page[limit]": page_limit,
        "page[offset]": page_offset,
    }
    return await mbta_get("/shapes", params=params)


# -----------------------------
# Optional OTP plan endpoint (not required)
# -----------------------------

@app.get("/route")
async def route_plan(
    from_lat: float = Query(...),
    from_lng: float = Query(...),
    to_lat: float = Query(...),
    to_lng: float = Query(...),
    arrive_by: bool = Query(False),
):
    now = datetime.now(ZoneInfo("America/New_York"))
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    url = f"{OTP_BASE_URL}/routers/default/plan"

    params = {
        "fromPlace": f"{from_lat},{from_lng}",
        "toPlace": f"{to_lat},{to_lng}",
        "date": date_str,
        "time": time_str,
        "mode": "TRANSIT,WALK",
        "arriveBy": "true" if arrive_by else "false",
        "numItineraries": 5,
        "walkReluctance": 2,
        "maxWalkDistance": 1600,
    }

    try:
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            return resp.json()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Trip planner error: {e}")


# -----------------------------
# Route options ranking (works with node-* by resolving to place-*)
# -----------------------------

async def mbta_get_one_stop(stop_id: str) -> Dict[str, Any]:
    payload = await mbta_get(f"/stops/{stop_id}", params=None)
    data = payload.get("data")
    if not data:
        raise HTTPException(status_code=404, detail=f"Stop not found: {stop_id}")
    return data


async def _resolve_stop_for_routes(stop_id: str) -> str:
    """
    MBTA /routes?filter[stop]=... often works best with parent station ids (place-...).
    If the user passes a node/platform/etc, map it to its parent_station.
    """
    try:
        stop = await mbta_get_one_stop(stop_id)
    except Exception:
        return stop_id

    rel = (stop.get("relationships") or {}).get("parent_station") or {}
    data = rel.get("data")

    if isinstance(data, dict):
        parent_id = data.get("id")
        if parent_id:
            return parent_id

    attrs = stop.get("attributes") or {}
    parent_attr = attrs.get("parent_station")
    if parent_attr:
        return parent_attr

    return stop_id


def _parse_depart(depart: Optional[str]) -> Optional[datetime]:
    # depart comes like "YYYY-MM-DDTHH:MM" from <input type="datetime-local">
    if not depart:
        return None
    try:
        return datetime.fromisoformat(depart)
    except Exception:
        return None


def _route_name(route_obj: Dict[str, Any]) -> str:
    attrs = route_obj.get("attributes") or {}
    return attrs.get("long_name") or attrs.get("short_name") or route_obj.get("id")


async def _stops_for_route(route_id: str, limit: int = 200) -> List[Dict[str, Any]]:
    payload = await mbta_get("/stops", params={"filter[route]": route_id, "page[limit]": limit})
    return payload.get("data", []) or []


async def _next_departure_at_stop(route_id: str, stop_id: str) -> Optional[datetime]:
    payload = await mbta_get(
        "/predictions",
        params={
            "filter[route]": route_id,
            "filter[stop]": stop_id,
            "sort": "departure_time",
            "page[limit]": 10,
        },
    )
    preds = payload.get("data", []) or []
    for p in preds:
        attrs = p.get("attributes") or {}
        t = attrs.get("departure_time") or attrs.get("arrival_time")
        if not t:
            continue
        try:
            return datetime.fromisoformat(t.replace("Z", "+00:00"))
        except Exception:
            continue
    return None


def _status_from_buffer_minutes(buffer_min: float) -> str:
    if buffer_min > 5:
        return "LIKELY"
    if buffer_min >= 0:
        return "RISKY"
    return "UNLIKELY"


@app.get("/route_options")
async def route_options(
    origin: str = Query(...),
    dest: str = Query(...),
    depart: Optional[str] = Query(None),
    limit: int = Query(5, ge=1, le=10),
):
    depart_dt = _parse_depart(depart)

    # ✅ Resolve node/platform -> place-... so routes filtering works
    origin_for_routes = await _resolve_stop_for_routes(origin)
    dest_for_routes = await _resolve_stop_for_routes(dest)

    # 1) routes serving origin & destination (using resolved ids)
    origin_routes_payload = await mbta_get("/routes", params={"filter[stop]": origin_for_routes, "page[limit]": 50})
    dest_routes_payload = await mbta_get("/routes", params={"filter[stop]": dest_for_routes, "page[limit]": 50})

    o_routes: Dict[str, Any] = {r["id"]: r for r in (origin_routes_payload.get("data") or [])}
    d_routes: Dict[str, Any] = {r["id"]: r for r in (dest_routes_payload.get("data") or [])}

    results: List[Dict[str, Any]] = []

    # 2) DIRECT routes
    for rid in (set(o_routes.keys()) & set(d_routes.keys())):
        results.append({
            "legs": [{"routeId": rid, "routeName": _route_name(o_routes[rid])}],
            "transferStop": None,
            "totalTime": None,
            "status": "LIKELY",
            "debug": {"type": "direct", "origin_for_routes": origin_for_routes, "dest_for_routes": dest_for_routes},
        })

    if len(results) >= limit:
        return results[:limit]

    # 3) ONE-TRANSFER routes
    stops_cache: Dict[str, List[Dict[str, Any]]] = {}
    stopset_cache: Dict[str, set] = {}

    async def get_stopset(route_id: str) -> set:
        if route_id in stopset_cache:
            return stopset_cache[route_id]
        stops = await _stops_for_route(route_id)
        stops_cache[route_id] = stops
        sset = set([s["id"] for s in stops if s.get("id")])
        stopset_cache[route_id] = sset
        return sset

    for d_id in d_routes.keys():
        await get_stopset(d_id)

    candidates: List[Tuple[float, Dict[str, Any]]] = []

    for o_id, o_route in o_routes.items():
        o_stopset = await get_stopset(o_id)

        for d_id, d_route in d_routes.items():
            if d_id == o_id:
                continue

            d_stopset = stopset_cache[d_id]
            transfer_ids = list(o_stopset & d_stopset)
            if not transfer_ids:
                continue

            transfer_ids = transfer_ids[:8]

            for transfer_stop_id in transfer_ids:
                tname = None
                for s in (stops_cache.get(o_id) or []):
                    if s.get("id") == transfer_stop_id:
                        tname = (s.get("attributes") or {}).get("name")
                        break

                o_next = await _next_departure_at_stop(o_id, transfer_stop_id)
                d_next = await _next_departure_at_stop(d_id, transfer_stop_id)
                if not o_next or not d_next:
                    continue

                if depart_dt:
                    try:
                        if o_next.replace(tzinfo=None) < depart_dt:
                            continue
                    except Exception:
                        pass

                buffer_min = (d_next - o_next).total_seconds() / 60.0 - 3.0
                status = _status_from_buffer_minutes(buffer_min)

                score = abs(buffer_min) if buffer_min >= 0 else 9999 + abs(buffer_min)

                candidates.append((
                    score,
                    {
                        "legs": [
                            {"routeId": o_id, "routeName": _route_name(o_route)},
                            {"routeId": d_id, "routeName": _route_name(d_route)},
                        ],
                        "transferStop": {"id": transfer_stop_id, "name": tname or transfer_stop_id},
                        "totalTime": None,
                        "status": status,
                        "debug": {
                            "type": "transfer",
                            "bufferMinutes": round(buffer_min, 2),
                            "oNext": o_next.isoformat(),
                            "dNext": d_next.isoformat(),
                            "origin_for_routes": origin_for_routes,
                            "dest_for_routes": dest_for_routes,
                        },
                    }
                ))

    candidates.sort(key=lambda x: x[0])
    for _, item in candidates:
        results.append(item)
        if len(results) >= limit:
            break

    return results[:limit]
