"""PATCH /api/v1/connectors/{id} and /api/v1/sources/{id} — connector/source
enable-disable state, relayed here by the backend's Connector Overview API."""

from typing import Annotated

from fastapi import APIRouter, Depends

from api.dependencies import get_source_state_store
from api.schemas import (
    ConfigureConnectorRequest,
    ConfigureConnectorResponse,
    PatchSourcesRequest,
    PatchSourcesResponse,
)
from ingestion.source_state_store import SourceStateStore

router = APIRouter()


@router.patch(
    "/connectors/{connector_id}",
    response_model=ConfigureConnectorResponse,
    summary="Enable or disable a connector",
    description=(
        "Enables or disables an entire connector (e.g. 'github'). Disabled "
        "connectors are excluded from retrieval."
    ),
)
def configure_connector(
    connector_id: str,
    body: ConfigureConnectorRequest,
    store: Annotated[SourceStateStore, Depends(get_source_state_store)],
) -> ConfigureConnectorResponse:
    store.set_connector_enabled(connector_id, body.enabled)
    return ConfigureConnectorResponse(connector_id=connector_id, enabled=body.enabled)


@router.patch(
    "/sources/{connector_id}",
    response_model=PatchSourcesResponse,
    summary="Enable or disable sources of a connector",
    description=(
        "Enables or disables individual sources of a connector (e.g. one "
        "GitHub repo, keyed by 'owner/repo'). Disabled sources are excluded "
        "from retrieval."
    ),
)
def patch_sources(
    connector_id: str,
    body: PatchSourcesRequest,
    store: Annotated[SourceStateStore, Depends(get_source_state_store)],
) -> PatchSourcesResponse:
    store.set_sources_enabled(connector_id, body.sources)
    return PatchSourcesResponse(connector_id=connector_id, sources=body.sources)
