"""Entity resolution REST endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from silicon_memory.entities import EntityResolver
from silicon_memory.llm.scheduler import LLMScheduler
from silicon_memory.server.dependencies import get_scheduler

logger = logging.getLogger(__name__)

router = APIRouter()


# --- Request/Response schemas ---


class RegisterAliasRequest(BaseModel):
    alias: str
    canonical_id: str
    entity_type: str


class RegisterAliasResponse(BaseModel):
    alias: str
    canonical_id: str
    stored: bool = True


class ResolveRequest(BaseModel):
    text: str


class ResolvedEntity(BaseModel):
    text: str
    canonical_id: str
    entity_type: str
    confidence: float
    span: list[int]
    rule_id: str | None = None


class ResolveResponse(BaseModel):
    resolved: list[ResolvedEntity]
    unresolved: list[str]


class BootstrapRequest(BaseModel):
    text: str = Field(description="Sample document text to learn entity patterns from")


class BootstrapResponse(BaseModel):
    detectors_created: int
    extractors_created: int
    aliases_discovered: int


class LearnResponse(BaseModel):
    rules_created: int


class RuleItem(BaseModel):
    id: str
    rule_type: str
    pattern: str
    entity_type: str | None = None
    description: str | None = None
    confidence: float | None = None


class RulesListResponse(BaseModel):
    detectors: list[RuleItem]
    extractors: list[RuleItem]
    total: int


# --- Dependency ---


def get_entity_resolver(request: Request) -> EntityResolver:
    """Get the EntityResolver from app state."""
    resolver = getattr(request.app.state, "entity_resolver", None)
    if resolver is None:
        raise HTTPException(status_code=503, detail="Entity resolver not initialized")
    return resolver


# --- Endpoints ---


@router.post("/entities/register")
async def register_alias(
    body: RegisterAliasRequest,
    resolver: EntityResolver = Depends(get_entity_resolver),
) -> RegisterAliasResponse:
    """Register an alias → canonical ID mapping."""
    await resolver.register_alias(body.alias, body.canonical_id, body.entity_type)
    return RegisterAliasResponse(alias=body.alias, canonical_id=body.canonical_id)


@router.post("/entities/resolve")
async def resolve_text(
    body: ResolveRequest,
    resolver: EntityResolver = Depends(get_entity_resolver),
) -> ResolveResponse:
    """Resolve entity references in text."""
    result = await resolver.resolve(body.text)
    return ResolveResponse(
        resolved=[
            ResolvedEntity(
                text=r.text,
                canonical_id=r.canonical_id,
                entity_type=r.entity_type,
                confidence=r.confidence,
                span=list(r.span),
                rule_id=r.rule_id,
            )
            for r in result.resolved
        ],
        unresolved=result.unresolved,
    )


@router.post("/entities/bootstrap")
async def bootstrap(
    body: BootstrapRequest,
    resolver: EntityResolver = Depends(get_entity_resolver),
    scheduler: LLMScheduler = Depends(get_scheduler),
) -> BootstrapResponse:
    """Bootstrap entity rules from a sample document using LLM."""
    from silicon_memory.entities.learner import RuleLearner

    learner = RuleLearner(llm=scheduler)
    detectors, extractors, aliases = await learner.bootstrap(body.text)

    for d in detectors:
        resolver.add_detector(d)
    for e in extractors:
        resolver.add_extractor(e)
    for short_form, long_form in aliases:
        await resolver.register_alias(short_form, long_form, "alias")

    return BootstrapResponse(
        detectors_created=len(detectors),
        extractors_created=len(extractors),
        aliases_discovered=len(aliases),
    )


@router.post("/entities/learn")
async def learn_rules(
    resolver: EntityResolver = Depends(get_entity_resolver),
    scheduler: LLMScheduler = Depends(get_scheduler),
) -> LearnResponse:
    """Generate rules from accumulated unresolved entities."""
    from silicon_memory.entities.learner import RuleLearner

    resolver._learner = RuleLearner(llm=scheduler)
    count = await resolver.learn_rules()
    return LearnResponse(rules_created=count)


@router.get("/entities/rules")
async def list_rules(
    resolver: EntityResolver = Depends(get_entity_resolver),
) -> RulesListResponse:
    """List all detector and extractor rules."""
    detectors = [
        RuleItem(id=d.id, rule_type="detector", pattern=d.pattern, description=d.description)
        for d in resolver.rules._detectors
    ]
    extractors = [
        RuleItem(
            id=e.id, rule_type="extractor", pattern=e.pattern,
            entity_type=e.entity_type, confidence=e.confidence,
        )
        for e in resolver.rules._extractors
    ]
    return RulesListResponse(
        detectors=detectors,
        extractors=extractors,
        total=len(detectors) + len(extractors),
    )
