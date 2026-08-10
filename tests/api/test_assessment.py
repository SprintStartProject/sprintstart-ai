import json
from collections.abc import Callable, Generator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from api.app import app
from api.dependencies import get_llm
from llm.base import Message
from llm.errors import LLMUnavailableError
from tests.stubs.llm import StubLLMClient

_URL = "/api/v1/onboarding/assessment/turn"

_CANDIDATES = [
    {"key": "kotlin", "label": "Kotlin", "description": "", "role_weight": 1.0},
    {
        "key": "jpa-persistence",
        "label": "JPA persistence",
        "description": "",
        "role_weight": 1.0,
    },
]


def _request(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "candidate_competencies": _CANDIDATES,
        "repo_signal": {
            "languages": ["kotlin"],
            "frameworks": ["spring-boot"],
            "notable": [],
        },
        "history": [],
        "turn": 0,
        "max_turns": 6,
        "must_finish": False,
    }
    body.update(overrides)
    return body


def _stub(payload: dict[str, Any]) -> Callable[[], StubLLMClient]:
    """Zero-arg factory for a StubLLMClient whose generate() returns payload as JSON.

    Must be zero-arg (not the class itself) since FastAPI's dependency_overrides
    introspects the override callable's signature -- passing the class directly
    would expose StubLLMClient.__init__'s params (e.g. embedding: list[float]) as
    if they were request body fields.
    """

    class _Stub(StubLLMClient):
        def generate(
            self, messages: list[Message], *, temperature: float | None = None
        ) -> str:
            return json.dumps(payload)

    return lambda: _Stub()


@pytest.fixture
def client() -> Generator[tuple[TestClient, StubLLMClient], Any, None]:
    llm = StubLLMClient()

    app.dependency_overrides[get_llm] = lambda: llm

    yield TestClient(app), llm

    app.dependency_overrides.clear()


def test_interviewing_turn_round_trips(client: tuple[TestClient, StubLLMClient]):
    http_client, _ = client
    payload = {
        "done": False,
        "question": "Walk me through adding a field to an entity.",
        "targets": ["kotlin", "jpa-persistence"],
        "coverage": [
            {"key": "kotlin", "level": None, "confidence": None},
            {"key": "jpa-persistence", "level": None, "confidence": None},
        ],
    }
    app.dependency_overrides[get_llm] = _stub(payload)

    response = http_client.post(_URL, json=_request())

    assert response.status_code == 200
    body = response.json()
    assert body["done"] is False
    assert body["question"] == payload["question"]
    assert set(body["targets"]) == {"kotlin", "jpa-persistence"}
    assert {c["key"] for c in body["coverage"]} == {"kotlin", "jpa-persistence"}


def test_finished_turn_covers_every_candidate(client: tuple[TestClient, StubLLMClient]):
    http_client, _ = client
    payload = {
        "done": True,
        "assessments": [
            {
                "key": "kotlin",
                "level": "advanced",
                "confidence": 0.8,
                "evidence": "Discussed null-safety tradeoffs unprompted.",
            },
            {
                "key": "jpa-persistence",
                "level": "intermediate",
                "confidence": 0.6,
                "evidence": "Named cascade types correctly.",
            },
        ],
    }
    app.dependency_overrides[get_llm] = _stub(payload)

    response = http_client.post(_URL, json=_request(turn=3))

    assert response.status_code == 200
    body = response.json()
    assert body["done"] is True
    assessments = {a["key"]: a for a in body["assessments"]}
    assert assessments.keys() == {"kotlin", "jpa-persistence"}
    assert assessments["kotlin"]["level"] == "advanced"
    assert assessments["jpa-persistence"]["level"] == "intermediate"


def test_must_finish_always_finishes_even_if_model_disagrees(
    client: tuple[TestClient, StubLLMClient],
):
    http_client, _ = client
    payload = {
        "done": False,
        "question": "One more thing...",
        "targets": ["kotlin"],
    }
    app.dependency_overrides[get_llm] = _stub(payload)

    response = http_client.post(_URL, json=_request(turn=6, must_finish=True))

    assert response.status_code == 200
    body = response.json()
    assert body["done"] is True
    assert {a["key"] for a in body["assessments"]} == {"kotlin", "jpa-persistence"}
    jpa = next(a for a in body["assessments"] if a["key"] == "jpa-persistence")
    # Probed but not assessed is "we asked and saw nothing" -- not "they are early".
    assert jpa["level"] == "none"
    assert jpa["confidence"] == 0.0
    assert jpa["evidence"] == "no signal"


def test_never_emits_keys_outside_candidates(client: tuple[TestClient, StubLLMClient]):
    http_client, _ = client
    payload = {
        "done": False,
        "question": "q",
        "targets": ["kotlin", "rust"],
        "coverage": [
            {"key": "kotlin", "level": None, "confidence": None},
            {"key": "rust", "level": None, "confidence": None},
        ],
    }
    app.dependency_overrides[get_llm] = _stub(payload)

    response = http_client.post(_URL, json=_request())

    assert response.status_code == 200
    body = response.json()
    assert body["targets"] == ["kotlin"]
    assert {c["key"] for c in body["coverage"]} == {"kotlin"}


def test_an_unreadable_level_is_not_credited(
    # ⚠️ It must land on "none", never "beginner". Only rank 0 is filtered out
    # downstream, so crediting a competency on the strength of a parse failure
    # shows up everywhere as "they have this".
    client: tuple[TestClient, StubLLMClient],
):
    http_client, _ = client
    payload = {
        "done": True,
        "assessments": [
            {"key": "kotlin", "level": "guru", "confidence": 0.9, "evidence": "e"},
            {
                "key": "jpa-persistence",
                "level": "intermediate",
                "confidence": 0.5,
                "evidence": "e",
            },
        ],
    }
    app.dependency_overrides[get_llm] = _stub(payload)

    # Past MIN_ASSESSMENT_TURNS so this exercises normalization, not the coverage floor.
    response = http_client.post(_URL, json=_request(turn=3))

    assert response.status_code == 200
    kotlin = next(a for a in response.json()["assessments"] if a["key"] == "kotlin")
    assert kotlin["level"] == "none"


def test_invalid_json_degrades_to_finalize_with_defaults_once_finishing_is_valid(
    client: tuple[TestClient, StubLLMClient],
):
    http_client, _ = client

    class GarbageLLM(StubLLMClient):
        def generate(
            self, messages: list[Message], *, temperature: float | None = None
        ) -> str:
            return "not json at all"

    app.dependency_overrides[get_llm] = lambda: GarbageLLM()

    response = http_client.post(_URL, json=_request(turn=6, must_finish=True))

    assert response.status_code == 200
    body = response.json()
    assert body["done"] is True
    assert {a["key"] for a in body["assessments"]} == {"kotlin", "jpa-persistence"}
    assert all(
        a["level"] == "none" and a["evidence"] == "no signal"
        for a in body["assessments"]
    )


def test_invalid_json_this_early_returns_503_rather_than_a_fabricated_placement(
    client: tuple[TestClient, StubLLMClient],
):
    """A garbled response on turn 0 can't be told apart from "the model tried to
    finish" -- both must be retryable, not a placement backed by nothing."""
    http_client, _ = client

    class GarbageLLM(StubLLMClient):
        def generate(
            self, messages: list[Message], *, temperature: float | None = None
        ) -> str:
            return "not json at all"

    app.dependency_overrides[get_llm] = lambda: GarbageLLM()

    response = http_client.post(_URL, json=_request(turn=0))

    assert response.status_code == 503


def test_llm_failure_returns_503(client: tuple[TestClient, StubLLMClient]):
    http_client, _ = client

    class FailingLLM(StubLLMClient):
        def generate(
            self, messages: list[Message], *, temperature: float | None = None
        ) -> str:
            raise LLMUnavailableError("http://localhost:11434")

    app.dependency_overrides[get_llm] = lambda: FailingLLM()

    response = http_client.post(_URL, json=_request())

    assert response.status_code == 503


def test_senior_transcript_skews_toward_higher_levels(
    client: tuple[TestClient, StubLLMClient],
):
    http_client, _ = client
    payload = {
        "done": True,
        "assessments": [
            {
                "key": "kotlin",
                "level": "expert",
                "confidence": 0.9,
                "evidence": "Discussed coroutine cancellation edge cases.",
            },
            {
                "key": "jpa-persistence",
                "level": "advanced",
                "confidence": 0.85,
                "evidence": "Explained N+1 query pitfalls unprompted.",
            },
        ],
    }
    app.dependency_overrides[get_llm] = _stub(payload)

    senior_answer = (
        "I'd add the column via a Flyway migration, update the entity with the "
        "right fetch strategy to avoid N+1s, and expose it through a DTO."
    )
    response = http_client.post(
        _URL,
        json=_request(
            history=[
                {"role": "assistant", "content": "Walk me through adding a field."},
                {"role": "user", "content": senior_answer},
            ],
            # Past MIN_ASSESSMENT_TURNS so this exercises level skew, not the
            # coverage floor.
            turn=3,
        ),
    )

    assert response.status_code == 200
    levels = {a["key"]: a["level"] for a in response.json()["assessments"]}
    assert levels["kotlin"] == "expert"
    assert levels["jpa-persistence"] == "advanced"


def test_junior_transcript_skews_toward_lower_levels(
    client: tuple[TestClient, StubLLMClient],
):
    http_client, _ = client
    payload = {
        "done": True,
        "assessments": [
            {
                "key": "kotlin",
                "level": "beginner",
                "confidence": 0.4,
                "evidence": "Unsure about null-safety operators.",
            },
            {
                "key": "jpa-persistence",
                "level": "beginner",
                "confidence": 0.3,
                "evidence": "No signal beyond 'I would look it up'.",
            },
        ],
    }
    app.dependency_overrides[get_llm] = _stub(payload)

    response = http_client.post(
        _URL,
        json=_request(
            history=[
                {"role": "assistant", "content": "Walk me through adding a field."},
                {"role": "user", "content": "I'm not sure, I'd probably look it up."},
            ],
            # Past MIN_ASSESSMENT_TURNS so this exercises level skew, not the
            # coverage floor.
            turn=3,
        ),
    )

    assert response.status_code == 200
    levels = {a["key"]: a["level"] for a in response.json()["assessments"]}
    assert levels["kotlin"] == "beginner"
    assert levels["jpa-persistence"] == "beginner"


def test_candidate_signal_reaches_the_interviewer_prompt(
    client: tuple[TestClient, StubLLMClient],
):
    """The consented involvement prior must actually be in front of the model."""
    http_client, _ = client
    captured: list[list[Message]] = []

    class _Capturing(StubLLMClient):
        def generate(
            self, messages: list[Message], *, temperature: float | None = None
        ) -> str:
            captured.append(messages)
            return json.dumps(
                {"done": False, "question": "q", "targets": [], "coverage": []}
            )

    app.dependency_overrides[get_llm] = lambda: _Capturing()

    response = http_client.post(
        "/api/v1/onboarding/assessment/turn",
        json=_request(
            candidate_signal={"signals": {"repo:owner/api": 9, "type:PULL_REQUEST": 9}}
        ),
    )

    assert response.status_code == 200
    user_message = captured[0][1]["content"]
    assert "repo:owner/api: 9" in user_message
    # Framed as a prior, not as evidence of skill -- the system prompt is what keeps the
    # model from assessing a competency off involvement alone.
    assert "weak prior only" in user_message
    system_message = captured[0][0]["content"]
    assert "NOT of proficiency" in system_message


def test_candidate_signal_is_optional(client: tuple[TestClient, StubLLMClient]):
    """A candidate who never consented still gets a normal interview."""
    http_client, _ = client
    app.dependency_overrides[get_llm] = _stub(
        {"done": False, "question": "q", "targets": [], "coverage": []}
    )

    response = http_client.post("/api/v1/onboarding/assessment/turn", json=_request())

    assert response.status_code == 200


def test_early_done_is_refused_and_the_interview_continues(
    client: tuple[TestClient, StubLLMClient],
):
    """One "I don't know" on turn 0 must not end the assessment.

    The live failure: a hire answered the first question with "i dont know, i am a
    beginner" and the interview finished immediately, placing every competency off a
    single non-answer.
    """
    http_client, _ = client
    responses = [
        {"done": True, "assessments": [{"key": "kotlin", "level": "beginner"}]},
        {
            "done": False,
            "question": "What have you built, even in a course project?",
            "targets": ["jpa-persistence"],
            "coverage": [],
        },
    ]

    class _Sequenced(StubLLMClient):
        def generate(
            self, messages: list[Message], *, temperature: float | None = None
        ) -> str:
            return json.dumps(responses.pop(0))

    app.dependency_overrides[get_llm] = lambda: _Sequenced()

    response = http_client.post(
        _URL,
        json=_request(
            turn=0,
            history=[
                {"role": "assistant", "content": "Walk me through a Kotlin service."},
                {"role": "user", "content": "i dont know, i am a beginner"},
            ],
        ),
    )

    assert response.status_code == 200
    body = response.json()
    assert body["done"] is False
    assert body["question"] == "What have you built, even in a course project?"
    assert body["targets"] == ["jpa-persistence"]


def test_the_retry_tells_the_model_a_weak_answer_is_not_grounds_to_finish(
    client: tuple[TestClient, StubLLMClient],
):
    http_client, _ = client
    captured: list[list[Message]] = []
    responses = [
        {"done": True, "assessments": []},
        {"done": False, "question": "q2", "targets": [], "coverage": []},
    ]

    class _Capturing(StubLLMClient):
        def generate(
            self, messages: list[Message], *, temperature: float | None = None
        ) -> str:
            captured.append(messages)
            return json.dumps(responses.pop(0))

    app.dependency_overrides[get_llm] = lambda: _Capturing()

    http_client.post(_URL, json=_request(turn=0))

    assert len(captured) == 2
    retry_instruction = captured[1][-1]["content"]
    assert "done=false" in retry_instruction
    assert "not about the rest" in retry_instruction


def test_a_model_that_insists_on_finishing_returns_503_instead_of_a_hollow_placement(
    client: tuple[TestClient, StubLLMClient],
):
    """A model that still says done=true after being told explicitly not to is
    exactly the failure mode the coverage floor exists to catch -- accepting it would
    place every unprobed candidate off zero evidence (turn 0 has no coverage
    history yet, so nothing here was ever asked about). Retryable, not fabricated."""
    http_client, _ = client
    app.dependency_overrides[get_llm] = _stub(
        {"done": True, "assessments": [{"key": "kotlin", "level": "advanced"}]}
    )

    response = http_client.post(_URL, json=_request(turn=0))

    assert response.status_code == 503


def test_an_unreachable_retry_returns_503_rather_than_a_fabricated_placement(
    client: tuple[TestClient, StubLLMClient],
):
    """The live failure this fixes: the model tries to finish on turn 0, the
    coverage-floor retry can't reach the LLM, and the old fallback silently accepted
    the early finish -- a "placement" backed by nothing, surfaced to the caller as a
    500/502 anyway since there was no question to return. Must be a clean, retryable
    503 instead."""
    http_client, _ = client
    calls = {"n": 0}

    class _FailsOnRetry(StubLLMClient):
        def generate(
            self, messages: list[Message], *, temperature: float | None = None
        ) -> str:
            calls["n"] += 1
            if calls["n"] > 1:
                raise LLMUnavailableError("down")
            return json.dumps(
                {"done": True, "assessments": [{"key": "kotlin", "level": "beginner"}]}
            )

    app.dependency_overrides[get_llm] = lambda: _FailsOnRetry()

    response = http_client.post(_URL, json=_request(turn=0))

    assert response.status_code == 503


def test_a_late_turn_may_still_finish_on_its_own(
    client: tuple[TestClient, StubLLMClient],
):
    """The floor applies to early turns only -- it must not force a full-length
    interview on somebody the model has genuinely finished placing."""
    http_client, _ = client
    calls = {"n": 0}

    class _Counting(StubLLMClient):
        def generate(
            self, messages: list[Message], *, temperature: float | None = None
        ) -> str:
            calls["n"] += 1
            return json.dumps(
                {"done": True, "assessments": [{"key": "kotlin", "level": "advanced"}]}
            )

    app.dependency_overrides[get_llm] = lambda: _Counting()

    response = http_client.post(_URL, json=_request(turn=4))

    assert response.json()["done"] is True
    assert calls["n"] == 1, "a late finish must not trigger the retry"


def test_must_finish_is_never_overridden_by_the_floor(
    client: tuple[TestClient, StubLLMClient],
):
    http_client, _ = client
    calls = {"n": 0}

    class _Counting(StubLLMClient):
        def generate(
            self, messages: list[Message], *, temperature: float | None = None
        ) -> str:
            calls["n"] += 1
            return json.dumps({"done": True, "assessments": []})

    app.dependency_overrides[get_llm] = lambda: _Counting()

    response = http_client.post(_URL, json=_request(turn=0, must_finish=True))

    assert response.json()["done"] is True
    assert calls["n"] == 1


def test_completion_is_refused_while_a_candidate_has_never_been_probed(
    client: tuple[TestClient, StubLLMClient],
):
    """The issue this closes: 10 candidates, 2 assessed, done=true after one
    "I don't know". A turn count cannot catch that; coverage can."""
    http_client, _ = client
    responses = [
        {"done": True, "assessments": []},
        {"done": False, "question": "And JPA?", "targets": ["jpa-persistence"]},
    ]

    class _Sequenced(StubLLMClient):
        def generate(
            self, messages: list[Message], *, temperature: float | None = None
        ) -> str:
            return json.dumps(responses.pop(0))

    app.dependency_overrides[get_llm] = lambda: _Sequenced()

    response = http_client.post(
        _URL,
        json=_request(
            turn=4,
            targets=[{"turn": 0, "keys": ["kotlin"]}],
        ),
    )

    # Turn 4 is past MIN_ASSESSMENT_TURNS, so only coverage can be refusing this.
    body = response.json()
    assert body["done"] is False
    assert body["targets"] == ["jpa-persistence"]


def test_completion_is_allowed_once_every_candidate_has_been_probed(
    client: tuple[TestClient, StubLLMClient],
):
    http_client, _ = client
    app.dependency_overrides[get_llm] = _stub(
        {
            "done": True,
            "assessments": [
                {
                    "key": "kotlin",
                    "level": "intermediate",
                    "confidence": 0.7,
                    "evidence": "explained coroutines",
                }
            ],
        }
    )

    response = http_client.post(
        _URL,
        json=_request(
            turn=1,
            targets=[{"turn": 0, "keys": ["kotlin", "jpa-persistence"]}],
        ),
    )

    body = response.json()
    assert body["done"] is True
    assert {a["key"] for a in body["assessments"]} == {"kotlin", "jpa-persistence"}


def test_a_key_that_was_never_asked_about_is_left_unplaced(
    client: tuple[TestClient, StubLLMClient],
):
    """ "Not asked" is not "asked and saw nothing" -- the caller records the
    latter as a level, so defaulting an unprobed key credits an assessment that
    never happened."""
    http_client, _ = client
    app.dependency_overrides[get_llm] = _stub({"done": True, "assessments": []})

    response = http_client.post(
        _URL,
        json=_request(
            turn=5,
            must_finish=True,
            targets=[{"turn": 0, "keys": ["kotlin"]}],
        ),
    )

    body = response.json()
    assert body["done"] is True
    assert [a["key"] for a in body["assessments"]] == ["kotlin"]


def test_the_turn_ceiling_still_ends_the_interview(
    client: tuple[TestClient, StubLLMClient],
):
    """Coverage gates completion but must never prevent it: a large candidate
    set cannot produce an endless interview."""
    http_client, _ = client
    app.dependency_overrides[get_llm] = _stub(
        {"done": False, "question": "one more?", "targets": []}
    )

    response = http_client.post(
        _URL,
        json=_request(
            turn=5,
            must_finish=True,
            targets=[{"turn": 0, "keys": ["kotlin"]}],
        ),
    )

    assert response.json()["done"] is True


def test_the_prompt_names_the_keys_that_have_not_been_probed(
    client: tuple[TestClient, StubLLMClient],
):
    http_client, _ = client
    captured: list[list[Message]] = []

    class _Capturing(StubLLMClient):
        def generate(
            self, messages: list[Message], *, temperature: float | None = None
        ) -> str:
            captured.append(messages)
            return json.dumps({"done": False, "question": "q", "targets": []})

    app.dependency_overrides[get_llm] = lambda: _Capturing()

    http_client.post(
        _URL,
        json=_request(turn=1, targets=[{"turn": 0, "keys": ["kotlin"]}]),
    )

    prompt = str(captured[0][-1]["content"])
    assert "Not probed yet (target these next): jpa-persistence" in prompt


def test_a_caller_sending_no_targets_still_gets_the_turn_floor(
    client: tuple[TestClient, StubLLMClient],
):
    """Without coverage information the service cannot tell probed from unprobed,
    so it falls back to the turn floor rather than blocking forever."""
    http_client, _ = client
    app.dependency_overrides[get_llm] = _stub(
        {
            "done": True,
            "assessments": [
                {
                    "key": "kotlin",
                    "level": "advanced",
                    "confidence": 0.8,
                    "evidence": "detailed",
                }
            ],
        }
    )

    response = http_client.post(_URL, json=_request(turn=4))

    body = response.json()
    assert body["done"] is True
    # No coverage information: every candidate is placed, as before.
    assert {a["key"] for a in body["assessments"]} == {"kotlin", "jpa-persistence"}


def test_a_confident_disclaimer_is_recorded_as_none_not_beginner(
    client: tuple[TestClient, StubLLMClient],
):
    """⚠️ The bug this exists to stop: being credited for saying you know nothing.

    A hire who says "I have never used Next.js" is giving a *confident* answer, so
    the interviewer reports high confidence. The backend's low-confidence floor
    therefore never fires -- the clearer the hire is, the more certain the model,
    and the more certainly the old contract credited them at beginner/rank 1.

    "none" is what makes the placement honest at the value, rather than relying on
    a downstream floor to rescue it.
    """
    http_client, _ = client
    payload = {
        "done": True,
        "assessments": [
            {
                "key": "kotlin",
                "level": "none",
                "confidence": 0.95,
                "evidence": "Says they have never used it.",
            },
            {
                "key": "jpa-persistence",
                "level": "intermediate",
                "confidence": 0.6,
                "evidence": "e",
            },
        ],
    }
    app.dependency_overrides[get_llm] = _stub(payload)

    response = http_client.post(_URL, json=_request(turn=3))

    assert response.status_code == 200
    kotlin = next(a for a in response.json()["assessments"] if a["key"] == "kotlin")
    assert kotlin["level"] == "none"
    # High confidence is preserved: the model is sure, and the value carries the
    # meaning. Nothing downstream should need the confidence to know it is not held.
    assert kotlin["confidence"] == 0.95


def test_the_real_levels_still_survive_normalization(
    client: tuple[TestClient, StubLLMClient],
):
    """ "none" must not swallow the scale it sits beside."""
    http_client, _ = client
    payload = {
        "done": True,
        "assessments": [
            {"key": "kotlin", "level": "EXPERT", "confidence": 0.9, "evidence": "e"},
            {
                "key": "jpa-persistence",
                "level": "  Advanced ",
                "confidence": 0.8,
                "evidence": "e",
            },
        ],
    }
    app.dependency_overrides[get_llm] = _stub(payload)

    response = http_client.post(_URL, json=_request(turn=3))

    assert response.status_code == 200
    levels = {a["key"]: a["level"] for a in response.json()["assessments"]}
    assert levels["kotlin"] == "expert"
    assert levels["jpa-persistence"] == "advanced"
