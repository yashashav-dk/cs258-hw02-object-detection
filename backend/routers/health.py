"""Health check endpoint."""

from fastapi import APIRouter

from backend.schemas.response import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    gpu_name = None
    gpu_mem_used = None
    gpu_mem_total = None

    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem_used = int(torch.cuda.memory_allocated(0) / (1024 * 1024))
            gpu_mem_total = int(torch.cuda.get_device_properties(0).total_mem / (1024 * 1024))
    except ImportError:
        pass

    return HealthResponse(
        gpu=gpu_name,
        gpu_memory_used_mb=gpu_mem_used,
        gpu_memory_total_mb=gpu_mem_total,
    )
