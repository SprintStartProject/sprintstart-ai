from typing import TYPE_CHECKING, Annotated, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic.alias_generators import to_camel

if TYPE_CHECKING:
    from onboarding.graph_models import ActiveCompetency, TombstonedCompetency
    from onboarding.models import Blueprint, PersonProfile
    from onboarding.verification import ArtifactEvidence


class IngestRequest(BaseModel):
    artifact_id: Annotated[
        str,
        Field(
            min_length=1,
            description=(
                "Stable identifier for the source document. "
                "Re-ingesting with the same artifact_id replaces all existing chunks."
            ),
            examples=["sprint-42-retro"],
        ),
    ]
    filename: Annotated[
        str,
        Field(
            min_length=1,
            description="Original filename, used in citations.",
            examples=["retro.md"],
        ),
    ]
    content: str = Field(
        description=(
            "Document content as a string. "
            "For text-based files (.txt, .md, .json, .yaml, .toml) send the raw text. "
            "For image files (.png, .jpg, .jpeg, .gif, .webp, .bmp) send the file "
            "as a standard base64-encoded string. "
            "If a vision model is not configured, image chunks are silently skipped "
            "and chunk_count will be 0."
        )
    )
    source_role: Literal["primary", "test"] | None = Field(
        default=None,
        description=(
            "Role of this document in the corpus. 'test' marks test code and "
            "test fixtures/sample data — still searchable, but excluded from "
            "onboarding grounding. Defaults to auto-detection from the filename."
        ),
        examples=["primary"],
    )
    semantic_boundaries: bool = Field(
        default=False,
        description=(
            "Only affects text and PDF content. When true, an LLM chooses "
            "chunk boundaries based on semantic coherence (topic shifts, "
            "section boundaries) instead of the default character-length "
            "accumulation. Falls back to the default chunker if the "
            "content is too large for the LLM or the LLM output is "
            "invalid. Independently toggleable from 'contextualize'."
        ),
    )
    contextualize: bool = Field(
        default=False,
        description=(
            "Only affects text and PDF content. When true, an LLM flags "
            "which chunks would benefit from a short situating context "
            "block (Anthropic-style Contextual Retrieval) and prepends it "
            "to their content; self-contained chunks are left untouched. "
            "Falls back to the default chunker if the content is too "
            "large for the LLM or the LLM output is invalid. "
            "Independently toggleable from 'semantic_boundaries'."
        ),
    )

    @field_validator("filename")
    @classmethod
    def filename_has_no_path_separators(cls, v: str) -> str:
        if "/" in v or "\\" in v:
            raise ValueError("filename must not contain path separators")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "artifact_id": "sprint-42-retro",
                "filename": "retro.md",
                "content": "# Retro\n## What went well\nGood collaboration...",
            }
        }
    }


class IngestArtifactResponse(BaseModel):
    id: str = Field(description="Artifact identifier.")
    filename: str = Field(description="Original source filename.")
    content_type: str = Field(description="Detected or inferred content type.")
    source_type: str = Field(description="Source type, e.g. file, text, or url.")
    size_bytes: int = Field(description="Size of the ingested content in bytes.")
    chunk_count: int = Field(description="Number of chunks created for this artifact.")
    status: str = Field(
        description="Ingestion status, e.g. processing, completed, or failed."
    )
    created_at: str = Field(description="ISO timestamp when ingestion started.")
    updated_at: str = Field(description="ISO timestamp when ingestion last changed.")
    error_message: str | None = Field(
        default=None,
        description="Failure reason if ingestion failed.",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "id": "sprint-42-retro",
                "filename": "retro.md",
                "content_type": "text/markdown",
                "source_type": "file",
                "size_bytes": 1234,
                "chunk_count": 2,
                "status": "completed",
                "created_at": "2026-06-21T12:00:00+00:00",
                "updated_at": "2026-06-21T12:00:01+00:00",
                "error_message": None,
            }
        }
    }


class IngestChunkResponse(BaseModel):
    id: str = Field(description="Chunk identifier.")
    artifact_id: str = Field(description="Artifact this chunk belongs to.")
    filename: str = Field(description="Original source filename.")
    text: str = Field(description="Stored chunk text.")
    chunk_index: int = Field(description="Position of this chunk within the artifact.")
    vector_store_id: str = Field(description="Identifier used in the vector store.")
    kind: str = Field(description="Chunk kind, e.g. text, code, pdf, or image.")

    model_config = {
        "json_schema_extra": {
            "example": {
                "id": "chunk-1",
                "artifact_id": "sprint-42-retro",
                "filename": "retro.md",
                "text": "Good collaboration and faster CI feedback...",
                "chunk_index": 0,
                "vector_store_id": "chunk-1",
                "kind": "text",
            }
        }
    }


class IngestResponse(BaseModel):
    artifact_id: str = Field(description="Created or updated artifact identifier.")
    chunk_count: int = Field(
        description=(
            "Number of chunks stored. 0 indicates the file was recognised "
            "but produced no storable content, e.g. an image file when "
            "no vision model is configured."
        )
    )
    artifact: IngestArtifactResponse
    chunks: list[IngestChunkResponse]

    model_config = {
        "json_schema_extra": {
            "example": {
                "artifact_id": "sprint-42-retro",
                "chunk_count": 2,
                "artifact": {
                    "id": "sprint-42-retro",
                    "filename": "retro.md",
                    "content_type": "text/markdown",
                    "source_type": "file",
                    "size_bytes": 1234,
                    "chunk_count": 2,
                    "status": "completed",
                    "created_at": "2026-06-21T12:00:00+00:00",
                    "updated_at": "2026-06-21T12:00:01+00:00",
                    "error_message": None,
                },
                "chunks": [
                    {
                        "id": "chunk-1",
                        "artifact_id": "sprint-42-retro",
                        "filename": "retro.md",
                        "text": "Good collaboration and faster CI feedback...",
                        "chunk_index": 0,
                        "vector_store_id": "chunk-1",
                        "kind": "text",
                    }
                ],
            }
        }
    }


class HistoryEntry(BaseModel):
    role: Literal["user", "assistant"] = Field(description="Who produced this message.")
    content: str = Field(description="Text content of the message.")

    model_config = {
        "json_schema_extra": {
            "example": {
                "role": "user",
                "content": "What were the main blockers in sprint 42?",
            }
        }
    }


SourceSystemValue = Literal["GITHUB", "JIRA", "UPLOAD"]


def _empty_history() -> list[HistoryEntry]:
    return []


class ChatFilters(BaseModel):
    source_systems: list[SourceSystemValue] | None = Field(
        default=None,
        description="Optional source systems to include. Empty or missing means all.",
    )
    time_from: str | None = Field(
        default=None,
        description="Optional inclusive lower bound as ISO-8601 timestamp.",
    )
    time_to: str | None = Field(
        default=None,
        description="Optional inclusive upper bound as ISO-8601 timestamp.",
    )

    @field_validator("source_systems", mode="before")
    @classmethod
    def normalize_source_systems(cls, value: object) -> object:
        if value is None:
            return None

        if not isinstance(value, list):
            return value

        items = cast(list[object], value)
        return [str(item).upper() for item in items]


class ChatRequest(BaseModel):
    question: str = Field(examples=["What changed in the auth implementation?"])
    history: list[HistoryEntry] = Field(default_factory=_empty_history)
    filters: ChatFilters | None = None

    @model_validator(mode="before")
    @classmethod
    def accept_legacy_chat_fields(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data

        raw_data = cast(dict[object, object], data)
        updated: dict[str, object] = {}

        for key, value in raw_data.items():
            updated[str(key)] = value

        if "question" not in updated and "prompt" in updated:
            updated["question"] = updated["prompt"]

        if "history" not in updated and "context" in updated:
            updated["history"] = updated["context"]

        return updated


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    detail: str | None = None


class TitleRequest(BaseModel):
    prompt: Annotated[
        str,
        Field(
            min_length=1,
            description="The input prompt used to generate the title",
            examples=["What are the main differences between REST and GraphQL?"],
        ),
    ]
    max_length: Annotated[
        int,
        Field(ge=1, le=200, description="Maximum title length"),
    ] = 60

    model_config = {
        "json_schema_extra": {
            "example": {
                "prompt": "What are the main differences between REST and GraphQL?",
                "max_length": 60,
            }
        }
    }

    @field_validator("prompt")
    @classmethod
    def prompt_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("prompt cannot be blank")
        return value


class TitleResponse(BaseModel):
    title: str = Field(description="Generated title based on the provided prompt.")

    model_config = {
        "json_schema_extra": {"example": {"title": "REST vs GraphQL: key differences"}}
    }


class ValidationErrorResponse(BaseModel):
    detail: str


class GradeAnswerItem(BaseModel):
    id: str = Field(
        description="Correlation id for this answer (the backend's questionId)."
    )
    question: str = Field(description="The short-text question being graded.")
    reference_answer: str = Field(description="The authored reference answer.")
    user_answer: str = Field(description="The user's submitted answer.")


class GradeAnswersRequest(BaseModel):
    answers: list[GradeAnswerItem] = Field(default_factory=list[GradeAnswerItem])


class GradeAnswerResult(BaseModel):
    id: str = Field(description="Correlation id matching the request item.")
    correct: bool = Field(description="Whether the answer is semantically correct.")
    confidence: float | None = Field(
        default=None, ge=0, le=1, description="Optional confidence score, 0..1."
    )
    feedback: str = Field(description="Short feedback shown to the user.")


class GradeAnswersResponse(BaseModel):
    results: list[GradeAnswerResult] = Field(default_factory=list[GradeAnswerResult])


class VectorDbChunkResponse(BaseModel):
    id: str = Field(description="Chunk identifier.")
    artifact_id: str = Field(description="Artifact/document this chunk belongs to.")
    filename: str = Field(description="Original source filename.")
    text: str = Field(description="Stored chunk text.")
    position: int | None = Field(
        default=None,
        description="Optional chunk position within the source artifact.",
    )
    kind: str = Field(description="Chunk kind, e.g. text, code, pdf, or image.")
    start_line: int | None = Field(
        default=None,
        description=(
            "1-based line the chunk starts on in the source file. "
            "Set for text/code sources; None for PDFs (see start_page)."
        ),
    )
    start_page: int | None = Field(
        default=None,
        description=(
            "1-based PDF page the chunk was extracted from. "
            "Set for PDF sources; None for text/code (see start_line)."
        ),
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "id": "chunk-1",
                "artifact_id": "artifact-123",
                "filename": "notes.md",
                "text": "Stored chunk text...",
                "position": 0,
                "kind": "text",
                "start_line": 12,
                "start_page": None,
            }
        }
    }


class VectorDbScoredChunkResponse(VectorDbChunkResponse):
    score: float = Field(description="Similarity score returned by vector search.")


class VectorDbChunkListResponse(BaseModel):
    items: list[VectorDbChunkResponse]
    limit: int
    offset: int
    total: int


class VectorDbStatusResponse(BaseModel):
    backend: str = Field(description="Configured vector store backend.")
    chunk_count: int = Field(description="Number of chunks currently stored.")

    model_config = {
        "json_schema_extra": {
            "example": {
                "backend": "chroma",
                "chunk_count": 128,
            }
        }
    }


class VectorDbSearchRequest(BaseModel):
    query: Annotated[
        str,
        Field(
            min_length=1,
            description="Query text to embed and search in the vector database.",
            examples=["Where is OLLAMA_EMBED_MODEL configured?"],
        ),
    ]
    top_k: Annotated[
        int,
        Field(ge=1, le=50, description="Maximum number of chunks to return."),
    ] = 5
    min_score: Annotated[
        float,
        Field(ge=0.0, description="Minimum similarity score to include."),
    ] = 0.0

    model_config = {
        "json_schema_extra": {
            "example": {
                "query": "Where is OLLAMA_EMBED_MODEL configured?",
                "top_k": 5,
                "min_score": 0.0,
            }
        }
    }


class VectorDbSearchResponse(BaseModel):
    items: list[VectorDbScoredChunkResponse]


class TokenEvent(BaseModel):
    type: Literal["token"]
    content: str = Field(
        description="A single token or short string fragment of the answer."
    )


class CitationEvent(BaseModel):
    type: Literal["citation"]
    artifact_id: str
    start_line: int | None = Field(
        default=None,
        description=(
            "1-based line the cited chunk starts on in the source file. "
            "Set for text/code sources; None for PDFs (see start_page)."
        ),
    )
    start_page: int | None = Field(
        default=None,
        description=(
            "1-based PDF page the cited chunk was extracted from. "
            "Set for PDF sources; None for text/code (see start_line)."
        ),
    )


class ToolUseEvent(BaseModel):
    type: Literal["tool_use"]
    name: Annotated[
        str,
        Field(
            description="Name of the invoked capability.",
            examples=["retrieve"],
        ),
    ]
    kind: Annotated[
        Literal["agent", "tool"],
        Field(
            description=(
                "Whether the invoked capability is a leaf 'tool' or a sub-'agent'."
            ),
        ),
    ]


class DoneEvent(BaseModel):
    type: Literal["done"]


class ErrorEvent(BaseModel):
    type: Literal["error"]
    message: str


class BlueprintStepSchema(BaseModel):
    id: str
    title: str
    description: str = ""
    requirement: str = "recommended"
    audience: list[str] = Field(default_factory=list)
    min_experience: str | None = None
    tags: list[str] = Field(default_factory=list)
    invariant: bool = False


class BlueprintProvenanceSchema(BaseModel):
    corpus_fingerprint: str | None = None
    generated_at: str | None = None
    model: str | None = None
    notes: list[str] = Field(default_factory=list)


class BlueprintSchema(BaseModel):
    scope: str
    version: str = "0"
    source: str = "authored"
    steps: list[BlueprintStepSchema] = []
    # Carried so the backend can round-trip it: ``corpus_fingerprint`` is what
    # lets a re-generation against an unchanged corpus short-circuit.
    provenance: BlueprintProvenanceSchema | None = None

    def to_model(self) -> "Blueprint":
        """Convert the wire schema into the internal Blueprint model."""
        from onboarding.models import Blueprint, BlueprintProvenance, BlueprintStep

        return Blueprint(
            scope=self.scope,
            version=self.version,
            source=self.source,  # type: ignore[arg-type]
            steps=[
                BlueprintStep(
                    id=s.id,
                    title=s.title,
                    description=s.description,
                    requirement=s.requirement,  # type: ignore[arg-type]
                    audience=s.audience,
                    min_experience=s.min_experience,
                    tags=s.tags,
                    invariant=s.invariant,
                )
                for s in self.steps
            ],
            provenance=(
                BlueprintProvenance(**self.provenance.model_dump())
                if self.provenance is not None
                else None
            ),
        )


class SkillAssessmentSchema(BaseModel):
    name: Annotated[
        str,
        Field(min_length=1, description="Skill tag, e.g. kotlin.", examples=["kotlin"]),
    ]
    level: Annotated[
        str,
        Field(
            default="beginner",
            description=(
                "Proficiency level: beginner, intermediate, advanced, expert. "
                "Case-insensitive; unknown values are handled gracefully."
            ),
            examples=["advanced"],
        ),
    ] = "beginner"


class OnboardingPathRequest(BaseModel):
    working_area: Annotated[
        str,
        Field(
            min_length=1,
            description="The person's working area, e.g. backend, frontend, devops.",
            examples=["backend"],
        ),
    ]
    skills: list[SkillAssessmentSchema] = Field(
        default_factory=list[SkillAssessmentSchema],
        description=(
            "Optional leveled skills ({name, level}); the backend supplies the "
            "user's skill assessments so proficiency drives personalization."
        ),
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Optional free-form tags used for step targeting.",
    )
    blueprints: list[BlueprintSchema] = Field(
        description=(
            "Active blueprints provided by the backend. The AI service is "
            "stateless — the backend owns blueprint persistence and must supply "
            "these on every request."
        ),
    )

    def to_profile(self) -> "PersonProfile":
        from onboarding.models import PersonProfile, SkillAssessment

        return PersonProfile(
            working_area=self.working_area,
            skills=[SkillAssessment(name=s.name, level=s.level) for s in self.skills],
            tags=self.tags,
        )

    model_config = {
        "json_schema_extra": {
            "example": {
                "working_area": "backend",
                "skills": [{"name": "kotlin", "level": "advanced"}],
                "tags": [],
                "blueprints": [
                    {
                        "scope": "global",
                        "version": "3",
                        "source": "generated",
                        "steps": [
                            {
                                "id": "step-abc123",
                                "title": "Set up development environment",
                                "description": "Install prerequisites",
                                "requirement": "required",
                            }
                        ],
                    }
                ],
            }
        }
    }


class GenerateBlueprintsRequest(BaseModel):
    scopes: list[str] | None = Field(
        default=None,
        description=(
            "Scopes to (re)generate, e.g. ['global', 'area:backend', 'area:frontend']. "
            "Omit to refresh 'global' plus any active blueprint scopes."
        ),
    )
    active: list[BlueprintSchema] = Field(
        default=[],
        description=(
            "The backend's currently-active blueprints. The AI service is "
            "stateless, so these drive idempotency and version numbering — pass "
            "them on every request."
        ),
    )


class StageEvent(BaseModel):
    type: Literal["stage"]
    name: Annotated[
        str,
        Field(
            description="The pipeline stage that just started.",
            examples=["retrieve"],
        ),
    ]


class PathEvent(BaseModel):
    type: Literal["path"]
    path: dict[str, object] = Field(
        description="The structured onboarding path (OnboardingPath model)."
    )
    path_yaml: str = Field(description="The onboarding path serialized to YAML.")
    quality: dict[str, object] = Field(
        description="The deterministic quality report for the path."
    )


# ── GitHub run batch ingest ───────────────────────────────────────────────────


class ArtifactRunIngestRequest(BaseModel):
    """One artifact from a completed GitHub ingestion run."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    artifact_id: str
    project_ids: list[str] = Field(
        default_factory=list[str],
        description=(
            "The projects this artifact belongs to. Several is ordinary -- a "
            "repository shared between two projects is one artifact serving both. "
            "Empty means unscoped, which stays searchable from every project: "
            "absent scope is not the same as excluded scope."
        ),
    )
    source_system: str | None = Field(
        default=None,
        alias="sourceSystem",
    )
    source_id: str
    source_url: str | None = None
    artifact_type: str
    title: str | None = None
    body_text: str | None = None
    mime: str | None = None
    language: str | None = None

    source_created_at: str | None = Field(
        default=None,
        alias="sourceCreatedAt",
        description="Original source creation timestamp, if known.",
    )
    source_updated_at: str | None = Field(
        default=None,
        alias="sourceUpdatedAt",
        description="Original source update timestamp, if known.",
    )

    state: str | None = Field(
        default=None,
        description="Issue state at the tracker ('OPEN'/'CLOSED'); null for non-issue.",
    )
    has_assignee: bool | None = Field(
        default=None,
        alias="hasAssignee",
        description=(
            "Whether somebody at the source is already assigned to this issue, or "
            "null when the connector cannot tell. Starter-work mining withholds "
            "an issue on a definite true -- taken work is not work a hire can "
            "pick up. Null is 'unknown', never 'nobody': GitHub issues have "
            "assignees this system does not ingest."
        ),
    )
    labels: list[str] = Field(
        default_factory=list,
        description="Issue labels (e.g. 'good first issue'); empty otherwise.",
    )


class RunArtifactsSyncRequest(BaseModel):
    """Batch payload sent by the backend after a GitHub ingestion run completes."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    artifacts_to_ingest: list[ArtifactRunIngestRequest]
    artifacts_to_deindex: list[str]


class ArtifactRunIngestResponse(BaseModel):
    artifact_id: str
    chunk_count: int
    status: Literal["completed", "failed"] = "completed"


class RunArtifactsSyncResponse(BaseModel):
    artifacts: list[ArtifactRunIngestResponse]


# ── Connector / source enable-disable ───────────────────────────────────────


class ConfigureConnectorRequest(BaseModel):
    enabled: bool = Field(description="Whether the connector should be enabled.")


class ConfigureConnectorResponse(BaseModel):
    connector_id: str
    enabled: bool


class PatchSourcesRequest(BaseModel):
    sources: dict[str, bool] = Field(
        description="Map of source id (e.g. 'owner/repo') to enabled status."
    )


class PatchSourcesResponse(BaseModel):
    connector_id: str
    sources: dict[str, bool]


class ArtifactSummaryRequest(BaseModel):
    previous_artifact_id: str | None = Field(
        default=None,
        alias="previousArtifactId",
        description="Optional previous artifact id for change summaries.",
    )
    max_chunks: int = Field(
        default=500,
        ge=1,
        le=2000,
        alias="maxChunks",
        description="Maximum number of chunks to use for summary generation.",
    )

    model_config = ConfigDict(populate_by_name=True)


# ── Knowledge-gaps (PM insights) ────────────────────────────────────────────


class KnowledgeGapSchema(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    component: str = Field(
        description="Component identifier derived from the ingestion index "
        "(e.g. owner/repo)."
    )
    missing_types: list[str] = Field(
        description="Expected documentation categories absent for this component."
    )
    present_types: list[str] = Field(
        description="Documentation categories the component already has."
    )
    last_updated: str = Field(
        description="ISO-8601 timestamp of the component's most recently updated "
        "artifact."
    )
    severity: Literal["high", "medium", "low"] = Field(
        description="Gap severity, from missing-critical-category count and staleness."
    )


class KnowledgeGapsResponse(BaseModel):
    gaps: list[KnowledgeGapSchema]


# ── FAQ grouping (PM insights) ──────────────────────────────────────────────


class FaqQuestionSchema(BaseModel):
    id: str = Field(description="Backend-assigned question identifier.")
    text: str = Field(description="The question's text.")

    model_config = {
        "json_schema_extra": {
            "example": {"id": "q_1", "text": "How do I get VPN access?"}
        }
    }


class FaqGroupRequest(BaseModel):
    questions: list[FaqQuestionSchema] = Field(
        description=(
            "Questions collected by the backend. The AI service is stateless "
            "and does not retain question history itself, so the full set to "
            "group is sent on every request."
        )
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "questions": [
                    {"id": "q_1", "text": "How do I get VPN access?"},
                    {"id": "q_2", "text": "Can someone enable VPN for me?"},
                ]
            }
        }
    }


class FaqDocumentSchema(BaseModel):
    id: str = Field(description="Knowledge-base artifact id for this document.")
    title: str = Field(description="Document title (filename).")
    source: str | None = Field(
        default=None,
        description="Origin system of the document, e.g. confluence, github.",
    )


class FaqGroupSchema(BaseModel):
    question: Annotated[
        str,
        Field(description="Representative question for the group, PII-redacted."),
    ]
    count: Annotated[
        int,
        Field(
            description=(
                "Total number of questions in the group. May be greater than "
                "len(questions), which is a redacted sample."
            )
        ),
    ]
    questions: Annotated[
        list[str],
        Field(description="PII-redacted sample of questions in the group."),
    ]
    documents: Annotated[
        list[FaqDocumentSchema],
        Field(description="Documents that answered the group's questions."),
    ]


class FaqGroupResponse(BaseModel):
    groups: list[FaqGroupSchema]

    model_config = {
        "json_schema_extra": {
            "example": {
                "groups": [
                    {
                        "question": "How do I get VPN access?",
                        "count": 14,
                        "questions": [
                            "How do I get VPN access?",
                            "Can someone enable VPN for me?",
                        ],
                        "documents": [
                            {
                                "id": "doc_001",
                                "title": "VPN Setup Guide",
                                "source": "confluence",
                            }
                        ],
                    }
                ]
            }
        }
    }


class ProposedCompetencySchema(BaseModel):
    key: str
    label: str
    description: str = ""
    kind: str
    area: str | None = None
    repo_ref: str | None = None


class TombstonedCompetencySchema(BaseModel):
    key: str
    label: str

    def to_model(self) -> "TombstonedCompetency":
        from onboarding.graph_models import TombstonedCompetency

        return TombstonedCompetency(key=self.key, label=self.label)


class GraphProvenanceSchema(BaseModel):
    corpus_fingerprint: str | None = None
    generated_at: str | None = None
    model: str | None = None
    notes: list[str] = Field(default_factory=list)


class ActiveCompetencySchema(BaseModel):
    key: str
    label: str
    description: str = ""
    kind: str
    area: str | None = None
    repo_ref: str | None = None

    def to_model(self) -> "ActiveCompetency":
        from onboarding.graph_models import ActiveCompetency

        return ActiveCompetency(
            key=self.key,
            label=self.label,
            description=self.description,
            kind=self.kind,  # type: ignore[arg-type]
            area=self.area,
            repo_ref=self.repo_ref,
        )


class GraphProposalOutcomeSchema(BaseModel):
    status: str
    competencies: list[ProposedCompetencySchema] = Field(
        default_factory=list[ProposedCompetencySchema]
    )
    provenance: GraphProvenanceSchema | None = None
    chunks_retrieved: int = 0
    notes: list[str] = Field(default_factory=list[str])


class GenerateCompetencyGraphRequest(BaseModel):
    active_competencies: list[ActiveCompetencySchema] = Field(
        default=[],
        description=(
            "The backend's current live competency vocabulary. Drives dedup -- "
            "an existing key is never re-proposed as new."
        ),
    )
    existing_areas: list[str] = Field(
        default=[],
        description=(
            "The grouping areas already in use. Sent so the generator reuses one "
            "instead of coining a synonym -- the area is free text, and a "
            "vocabulary split across 'auth', 'Authentication' and 'Auth & "
            "Identity' groups nothing. The backend normalises on write too; this "
            "is what lets the model choose rather than be corrected."
        ),
    )
    tombstoned_competencies: list[TombstonedCompetencySchema] = Field(
        default=[],
        description=(
            "Competencies a person deliberately removed. Blocked by key and by "
            "embedding similarity, and named in the prompt as rejected -- a "
            "deletion that only holds against the exact key leaks, because the "
            "competency returns next crawl under a rephrasing."
        ),
    )
    last_fingerprint: str | None = Field(
        default=None,
        description=(
            "The corpus fingerprint recorded from the caller's previous proposal "
            "run, if any. The AI service is stateless, so idempotency is driven "
            "by this rather than by state kept here -- there is no persisted "
            "'active proposal' the way a generated artifact's provenance carries one."
        ),
    )


class ProposeModuleRequest(BaseModel):
    competency_key: str = Field(description="The competency this module teaches.")
    competency_label: str
    competency_description: str = ""
    level: str = Field(
        default="beginner",
        description="Target level to teach to: beginner/intermediate/advanced/expert.",
    )
    last_fingerprint: str | None = Field(
        default=None,
        description=(
            "The corpus fingerprint recorded from the caller's previous proposal "
            "run for this competency, if any. Idempotency is per module, not "
            "corpus-wide -- modules are proposed one node at a time."
        ),
    )


class MineStarterWorkRequest(BaseModel):
    active_source_ids: list[str] = Field(
        default=[],
        description=(
            "Issues already in the backend's starter-work pool (proposed or "
            "approved). Drives dedup -- never re-proposed."
        ),
    )
    active_competency_keys: list[str] = Field(
        default=[],
        description=(
            "The backend's live competency graph keys, used to ground each "
            "task's competency tags. A tag outside this set is dropped, not "
            "invented; when empty, tags are kept as proposed."
        ),
    )
    last_fingerprint: str | None = Field(
        default=None,
        description=(
            "The corpus fingerprint recorded from the caller's previous "
            "mining run, if any."
        ),
    )


class AssembleOrientationRequest(BaseModel):
    task_title: str = Field(description="The task the packet orients somebody for.")
    task_body: str = ""
    labels: list[str] = Field(default_factory=list)
    touched_paths: list[str] = Field(
        default_factory=list,
        description=(
            "Repository paths the task is expected to touch, when known. Used to "
            "aim retrieval at the right part of the codebase, never asserted."
        ),
    )
    last_fingerprint: str | None = Field(
        default=None,
        description=(
            "The corpus fingerprint recorded when this task's packet was last "
            "assembled, if any. Idempotency is per task: an unchanged corpus "
            "yields `unchanged` so a cached packet can be served, and a moved "
            "corpus regenerates rather than describing code that changed."
        ),
    )


class AssembleDiagramRequest(BaseModel):
    subject: str = Field(
        description=(
            "The question the diagram answers -- 'how a request reaches the "
            "database'. This is the one part of a diagram a model chooses, and "
            "it only aims retrieval: every part that comes back is derived from "
            "the corpus and cited, so a subject nothing supports yields "
            "`skipped` rather than an invention."
        )
    )
    last_fingerprint: str | None = Field(
        default=None,
        description=(
            "The corpus fingerprint recorded when this subject was last drawn, "
            "if any. Idempotency is per subject: an unchanged corpus yields "
            "`unchanged` so a cached diagram can be served without a generation "
            "-- which is what keeps a card that hydrates on every board load "
            "from costing an LLM call every time."
        ),
    )


class FileDiffSchema(BaseModel):
    """One changed file's diff, budgeted backend-side."""

    path: str
    additions: int = 0
    deletions: int = 0
    patch: str | None = Field(
        default=None,
        description="None when GitHub reported no patch -- binary or too large.",
    )
    truncated: bool = Field(
        default=False,
        description=(
            "Whether the patch was cut, so a trimmed diff is not read as small."
        ),
    )


class ArtifactEvidenceSchema(BaseModel):
    pr_title: str = ""
    pr_body: str = ""
    pr_state: str = Field(
        default="", description="e.g. 'OPEN'/'MERGED'/'CLOSED'; informational only."
    )
    files_changed: list[str] = Field(default_factory=list)
    checks_passed: bool | None = Field(
        default=None, description="None when CI status is unknown/not reported."
    )
    commit_messages: list[str] = Field(default_factory=list)
    file_diffs: list[FileDiffSchema] = Field(
        default_factory=list[FileDiffSchema],
        description=(
            "The changed files' diffs. Filenames alone cannot separate a real fix "
            "from a whitespace edit to the right file."
        ),
    )
    omitted_file_count: int = Field(
        default=0,
        description=(
            "Changed files whose diff did not fit the budget -- sent so a partial "
            "diff is never mistaken for a complete one."
        ),
    )

    def to_model(self) -> "ArtifactEvidence":
        from onboarding.verification import ArtifactEvidence

        return ArtifactEvidence(**self.model_dump())


class VerifyRequest(BaseModel):
    type: str = Field(description="Grading type: knowledge/exact/attest/artifact.")
    question: str = ""
    answer: str = ""
    attempt_no: int = Field(default=1, ge=1)
    canonical_answer: str | None = Field(
        default=None, description="Required for 'exact' grading."
    )
    rubric: str | None = Field(
        default=None, description="Required for 'knowledge' and 'artifact' grading."
    )
    evidence: str = Field(
        default="",
        description=(
            "Grounded evidence backing the rubric (e.g. the lesson body), used "
            "only for 'knowledge' grading."
        ),
    )
    artifact_evidence: ArtifactEvidenceSchema | None = Field(
        default=None,
        description=(
            "Backend-gathered PR/repo state, used only for 'artifact' grading. "
            "Missing/empty is treated as no evidence yet."
        ),
    )
