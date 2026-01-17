# mongo_api.py
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

import mongo_repo

router = APIRouter(prefix="/mongo", tags=["mongo"])


class MongoEventIn(BaseModel):
    title: str
    starts_at: str  # keep as string like your Snowflake API
    venue: Optional[str] = None
    city: Optional[str] = None


@router.get("/health")
async def mongo_health():
    try:
        await run_in_threadpool(mongo_repo.ping)
        return {"ok": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/events")
async def mongo_insert_event(e: MongoEventIn):
    try:
        inserted_id = await run_in_threadpool(mongo_repo.insert_event, e.model_dump())
        return {"ok": True, "inserted_id": inserted_id}
    except Exception as ex:
        raise HTTPException(status_code=500, detail=str(ex))


@router.get("/events")
async def mongo_read_events(limit: int = Query(50, ge=1, le=1000)):
    try:
        docs = await run_in_threadpool(mongo_repo.read_events, limit)
        return {"ok": True, "events": docs}
    except Exception as ex:
        raise HTTPException(status_code=500, detail=str(ex))
