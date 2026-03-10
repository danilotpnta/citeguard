import uuid
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.core.monitoring import performance_tracker
from app.services.citeguard_service import CiteGuardService
from app.api.middleware.rate_limit import check_rate_limit
from app.api.dependencies import get_current_token
from app.core.logging import (
    get_logger,
    set_request_context,
    clear_request_context,
)
from app.models.schemas import (
    ReferenceResult,
    ReferenceStatus,
    VerificationStatus,
    VerifyRequest,
    VerifyResponse,
)
from app.models.token import Token

logger = get_logger(__name__)

router = APIRouter(tags=["verify"])

# In-memory job store. Will be replaced by a proper store if we go async.
_jobs: dict[str, VerifyResponse] = {}

# Initialize service (singleton)
citeguard_service = CiteGuardService()


@router.post("/verify", response_model=VerifyResponse)
async def verify_text(
    body: VerifyRequest,
    token: Token = Depends(check_rate_limit),
):
    """
    Submit text containing references to verify.
    Each reference is checked against scholarly databases.
    """
    job_id = f"job_{uuid.uuid4().hex[:12]}"
    set_request_context(request_id=job_id, user_id=token.company)

    try:
        # Call service layer (all business logic is there)
        with performance_tracker.track("verify_from_text_total"):
            results = await citeguard_service.verify_from_text(
                text=body.text,
                user_id=token.token_id,
            )
            # results = _stub_results(body.text)

        logger.info(f"Verification pipeline run successfully")

    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)

    verified = sum(1 for r in results if r.status == ReferenceStatus.VERIFIED)
    suspicious = sum(1 for r in results if r.status == ReferenceStatus.SUSPICIOUS)
    hallucinated = sum(1 for r in results if r.status == ReferenceStatus.HALLUCINATED)
    unresolved = sum(1 for r in results if r.status == ReferenceStatus.UNRESOLVED)

    response = VerifyResponse(
        job_id=job_id,
        status=VerificationStatus.COMPLETED,
        created_at=datetime.now(UTC),
        token_id=token.token_id,
        total_references=len(results),
        verified=verified,
        suspicious=suspicious,
        hallucinated=hallucinated,
        unresolved=unresolved,
        references=results,
    )

    _jobs[job_id] = response
    clear_request_context()
    return response


@router.post("/verify/upload", response_model=VerifyResponse)
async def verify_file(
    file: UploadFile = File(...),
    token: Token = Depends(check_rate_limit),
):
    """
    Upload a PDF, DOCX, or TXT file containing references to verify.
    """
    allowed_types = {
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "text/plain",
    }
    content_type = file.content_type or ""
    if content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {content_type}. Accepted: PDF, DOCX, TXT.",
        )

    content = await file.read()

    if len(content) > 5 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="File too large. Maximum size is 5MB.",
        )

    job_id = uuid.uuid4().hex[:16]

    # TODO: Phase 4 -- parse file with services/parser.py
    # TODO: Phase 5 -- replace stub with LangGraph pipeline call
    text = (
        content.decode("utf-8", errors="ignore") if content_type == "text/plain" else ""
    )
    results = _stub_results(text)

    verified = sum(1 for r in results if r.status == ReferenceStatus.VERIFIED)
    suspicious = sum(1 for r in results if r.status == ReferenceStatus.SUSPICIOUS)
    hallucinated = sum(1 for r in results if r.status == ReferenceStatus.HALLUCINATED)
    unresolved = sum(1 for r in results if r.status == ReferenceStatus.UNRESOLVED)

    response = VerifyResponse(
        job_id=job_id,
        status=VerificationStatus.COMPLETED,
        created_at=datetime.now(UTC),
        token_id=token.token_id,
        total_references=len(results),
        verified=verified,
        suspicious=suspicious,
        hallucinated=hallucinated,
        unresolved=unresolved,
        references=results,
    )

    _jobs[job_id] = response
    return response


@router.get("/verify/{job_id}", response_model=VerifyResponse)
async def get_result(
    job_id: str,
    token: Token = Depends(get_current_token),
):
    """
    Retrieve verification results by job ID.
    """
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    # Only allow the token that created the job to view it
    if job.token_id != token.token_id:
        raise HTTPException(
            status_code=403, detail="You do not have access to this job."
        )

    return job


def _stub_results(text: str) -> list[ReferenceResult]:
    """
    Placeholder that returns a fake result.
    Will be replaced by the LangGraph pipeline in Phase 5.
    """
    if not text.strip():
        return []

    return [
        ReferenceResult(
            raw_reference="[stub] No pipeline connected yet",
            title="Stub Reference",
            authors=["Pipeline, Not Yet"],
            year=2025,
            doi=None,
            status=ReferenceStatus.UNRESOLVED,
            confidence=0.0,
            sources_checked=[],
            match_details="LangGraph pipeline not yet connected. This is a placeholder response.",
        )
    ]
