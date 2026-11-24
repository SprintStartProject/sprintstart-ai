# Contracts

Machine-readable contracts between the SprintStart services. Each contract has one
owner — the repository that produces the data — and every consumer keeps an identical copy.

| Contract | Owner (producer) | Consumers | Producer test | Consumer test |
| --- | --- | --- | --- | --- |
| `agent-stream-events` — SSE of `POST /api/v1/onboarding/buddy/agent/stream` | sprintstart-ai | sprintstart-backend | `tests/contracts/test_agent_stream_contract.py` (ai) | `StreamContractTest` (backend) |
| `buddy-stream-events` — SSE of `POST /api/v1/onboarding/me/buddy/messages` and `/open/stream` | sprintstart-backend | sprintstart-frontend | `StreamContractTest` (backend) | `tests/unit/contracts/buddyStream.contract.test.ts` (frontend) |

Each contract is a JSON Schema (`<name>.schema.json`, one `oneOf` branch per event type) plus one example per
event type under `examples/<name>/`. Producer tests check that what the code really emits follows the schema;
consumer tests check that every example is read correctly by the code that reads it in production.

**Changing a contract** is an ADR trigger. Change the owner's schema and examples, then copy them to every
consumer in the same week; `bash scripts/quality/contracts-sync.sh` fails while a copy differs from the owner's
`dev` branch.
