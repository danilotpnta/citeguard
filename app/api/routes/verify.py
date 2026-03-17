import uuid
from datetime import UTC, datetime

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    UploadFile,
)
from app.models.token import Token
from app.api.dependencies import get_current_token
from app.core.monitoring import performance_tracker
from app.services.citeguard_service import CiteGuardService
from app.api.middleware.rate_limit import check_rate_limit
from app.models.constants import ContentType, UploadConfig
from app.models.schemas import VerifyRequest, VerifyResponse

from app.core.logging import (
    get_logger,
    set_request_context,
    clear_request_context,
)

router = APIRouter(tags=["verify"])

logger = get_logger(__name__)


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
    job_id = f"job_text_{uuid.uuid4().hex[:12]}"
    set_request_context(request_id=job_id, user_id=token.company)

    try:
        # Call service layer (all business logic is there)
        with performance_tracker.track("verify_from_text_total"):
            response = await citeguard_service.verify_from_text(
                raw_input=body.text,
                content_type="text",
                token_id=token.token_id,
            )

        logger.info(f"Verification pipeline run successfully")
        _jobs[job_id] = response
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise
    finally:
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
    try:
        content_type = ContentType.from_mime(file.content_type or "").short
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    logger.info(f"Processing filename {file.filename}")
    content = await file.read()

    if len(content) > UploadConfig.MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="File too large. Maximum size is 5MB.",
        )

    job_id = f"job_file_{uuid.uuid4().hex[:12]}"
    set_request_context(request_id=job_id, user_id=token.company)

    try:
        # Call service layer (all business logic is there)
        with performance_tracker.track("verify_from_file_total"):
            response = await citeguard_service.verify_from_file(
                raw_input=content,
                content_type=content_type,
                token_id=token.token_id,
                filename=file.filename,
            )

        logger.info(f"Verification pipeline run successfully")
        _jobs[job_id] = response
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise
    finally:
        clear_request_context()

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
