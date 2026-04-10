"""Models listing endpoint."""

from fastapi import APIRouter, Request

from backend.schemas.response import ModelsListResponse

router = APIRouter(prefix="/v1")


@router.get("/models", response_model=ModelsListResponse)
async def list_models(request: Request) -> ModelsListResponse:
    manager = request.app.state.model_manager
    return ModelsListResponse(data=manager.list_models())
