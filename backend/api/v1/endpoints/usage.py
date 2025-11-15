from typing import Optional

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.database import get_db
from backend.modules.applications.crud import FREE_TRIAL_REQUEST_LIMIT, get_ip_usage
from backend.modules.applications.models import IPUsage

router = APIRouter()


class FreeTrialUsageResponse(BaseModel):
    limit: int
    used: int
    remaining: int


def extract_client_ip(request: Request) -> Optional[str]:
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        parts = [ip.strip() for ip in forwarded_for.split(",")]
        if parts:
            return parts[0]

    real_ip = request.headers.get("x-real-ip")
    if real_ip:
        return real_ip.strip()

    if request.client:
        return request.client.host

    return None


def _parse_request_count(usage: Optional[IPUsage]) -> int:
    if not usage:
        return 0
    try:
        return int(usage.request_count or "0")
    except (ValueError, TypeError):
        return 0


@router.get("/free-trial", response_model=FreeTrialUsageResponse)
async def get_free_trial_usage(
    request: Request, db: AsyncSession = Depends(get_db)
) -> FreeTrialUsageResponse:
    client_ip = extract_client_ip(request)
    usage_record = await get_ip_usage(db, client_ip) if client_ip else None
    used = _parse_request_count(usage_record)
    remaining = max(FREE_TRIAL_REQUEST_LIMIT - used, 0)
    return FreeTrialUsageResponse(
        limit=FREE_TRIAL_REQUEST_LIMIT, used=used, remaining=remaining
    )
