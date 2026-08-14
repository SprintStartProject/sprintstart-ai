# FAQ grouping & category concept

Design record for stories #284 and #285 (epic #62, Analytics/Evaluation/Experimentation), from the
Sprint 6 review on 2026-08-12. It describes how the FAQ insight is structured, how it stays
current, and how it is kept readable as the question set grows. It spans three repositories, so it
lives here, where the classification itself does.

## The problem

The original FAQ was a **flat list of question groups, rebuilt on demand**. A PM pressed refresh,
the backend sent every question the project had ever asked to the AI service, one LLM pass
clustered all of them, and the result replaced the previous one.

Two things follow from that shape, and both were raised in the review:

1. **It cannot be current.** The rebuild's cost grows with the total number of questions, so it can
   never run per chat message. The panel is stale by default, and a PM has no way of knowing by how
   much.
2. **It cannot scale.** Recurring questions accumulate forever. Two hundred groups in one list is
   the "drowning in questions" problem the grouping was supposed to solve, arriving a little later.

## The structure

Two levels, deliberately:

| Level        | Means                                              | Example                                     |
| ------------ | -------------------------------------------------- | ------------------------------------------- |
| **Group**    | One recurring question — the same thing asked in different words | "How do I get VPN access?" / "Can someone enable VPN for me?" |
| **Category** | A topic bucket several groups share                | "Access & Accounts"                         |

A category is a plain label on the group, not an entity of its own. Groups are the thing with
identity, samples, documents and a count; a category is how they are filed. That keeps merging and
renaming categories a cheap operation — it re-labels rows — and avoids a second lifecycle to keep in
sync with the first.

Groups with no category exist and are shown as such ("Not yet categorised"). Two honest reasons:
rows written before categories existed, and questions the classifier could not place. Inventing a
topic for them would be worse than admitting there isn't one.

## Staying current (#284)

The rebuild is no longer the normal path. It is the fallback.

```
user asks the AI Buddy
  → chat stores the message, publishes ChatQuestionAskedEvent (async, fire-and-forget)
  → insights calls POST /insights/faq/classify with: the question
                                                   + existing categories (≤ ceiling)
                                                   + candidate groups (≤ candidate limit)
  → AI answers: relevant? · which category · existing group or a new one · redacted text
  → insights applies it, then enforces the structure limits
```

The essential property is that **the prompt is bounded by the FAQ's structure, not by its history**.
Filing the ten-thousandth question costs the same as filing the tenth. That is what makes it
affordable on every message, and it is the whole reason the incremental path exists next to the
rebuild rather than replacing it with a smaller rebuild.

Supporting decisions:

- **The event fires when the question is asked, not when it is answered.** The FAQ is about what
  people ask; it should not depend on the answer arriving.
- **The chat never waits for it.** A plain event listener runs on the publisher's thread, so the
  work is handed to a coroutine immediately. The FAQ may lag by a second; the chat may not.
- **Classification is idempotent** via the source message id. Events can be redelivered, and
  counting a message twice would quietly corrupt the frequency ranking the whole panel is ordered
  by.
- **Retrieval only runs for a new group.** An existing group already carries the documents that
  answer it.
- **Redaction is folded into the same call**, rather than a second round-trip per message.

### Why not exact

One question carries little context. Judged against a bounded candidate list, the classifier will
occasionally open a group that duplicates an existing one — most likely a group created minutes
ago, which has a count of one and would not survive a purely frequency-ranked candidate cut. The
candidate selection mitigates this (half the budget goes to the most-asked groups, half to the most
recently asked), but does not eliminate it.

The alternative — comparing every question against the full corpus — is exactly the cost this
design exists to avoid. So the drift is accepted and then **bounded**, by the consolidation passes
below. The system is self-healing rather than exact.

## Staying readable (#285)

Two ceilings, both configurable under `sprintstart.insights.faq`:

| Ceiling                   | Default | Crossed → |
| ------------------------- | ------- | --------- |
| `max-categories`          | 12      | `POST /insights/faq/categories/consolidate` proposes merges of related categories |
| `max-groups-per-category` | 20      | `POST /insights/faq/groups/merge` proposes merges of duplicate groups in that category |

Both are enforced by **folding things together, never by refusing an entry** — a limit must not
lose a question.

Both passes are cheap for the same reason the classification is: they send structure, not history.
Category consolidation sees category names and counts only — no question text at all. Group merging
sees one category's representative questions. That is why they can run on crossing a ceiling
instead of being scheduled nightly.

Order matters: category consolidation runs first, because it can move groups into the category the
new question landed in, and checking the group ceiling before that would miss the overflow it just
caused.

**The plans are proposals.** The AI service validates its own output (unknown names, a source
claimed twice, a target that is itself merged away), and the backend validates it again before
applying it, because the backend is the one applying it destructively. A plan that cannot be
applied safely is dropped, not guessed at. An empty plan is a legitimate answer: staying over the
limit beats merging distinct topics.

## Growing and shifting over time (#285)

A count alone cannot distinguish a topic that is picking up from one that was asked constantly a
year ago and never since — but to a PM deciding where documentation effort pays off, those mean
opposite things.

So every group and category also carries a **trend**: questions in the current window (default 14
days) against the window before it.

- `RISING` — asked more than before
- `STEADY` — asked about as often
- `FADING` — asked less, **or not at all in either window**

The last clause is the one that matters. Two empty windows are not a topic holding its level; they
are a topic nobody has asked about in a month.

Categories are ordered by **recent** volume rather than all-time count, which is what makes growing
topics surface and stale ones sink on their own, with no manual curation.

Trends are measured over the questions actually stored for a group. Every question arriving through
the live path is stored, so the numbers are exact for a FAQ maintained that way. Directly after a
full rebuild — which carries back only a redacted sample per group — they understate large groups
until live traffic accumulates again. The group's *recency* (`firstAskedAt` / `lastAskedAt`) stays
exact either way, because the rebuild returns the ids of every question it grouped and the backend
maps those back to the messages they were asked in.

## Knowledge gaps (#284)

The same "why should a PM press anything" argument applies, with a different trigger. Knowledge
gaps are derived entirely from what the AI service has indexed, so they go stale the moment a
repository is ingested.

The rescan therefore hangs off `ArtifactsIndexedEvent` — published once a run's artifacts are
**searchable**, not when the run merely finished. A scan started at the earlier point would query
the AI service before the new documents are in its corpus and dutifully report the gaps the
ingestion just closed.

Runs are debounced per project (default 60s), because connecting a repository produces a burst of
runs (files, commits, issues, pull requests) and each one arriving would otherwise trigger its own
full scan over the same corpus. While a rescan is pending or running the API says so, so the panel
can show "Scanning…" instead of stale numbers with no explanation.

## What the manual refresh is for now

It still exists, and the acceptance criteria require it to. But its meaning changed: it is no
longer how a PM sees new questions — it is how they throw the whole grouping away and rebuild it,
after the structure has drifted or the prompts have changed. The UI calls it "Rebuild grouping"
rather than "Refresh" for that reason.

The rebuild assigns categories too, so it cannot wipe the categorisation it is meant to repair.

## Configuration

```yaml
sprintstart:
  insights:
    faq:
      live-updates: true          # file questions as they are asked
      max-categories: 12          # ceiling before categories are consolidated
      max-groups-per-category: 20 # ceiling before duplicate groups are merged
      candidate-groups: 40        # groups one classification may consider
      sample-questions: 10        # sample questions shown on a group's detail view
      trend-window-days: 14       # window a trend is measured over
    knowledge-gaps:
      auto-refresh: true          # rescan once new documentation is indexed
      debounce-seconds: 60        # coalesce a burst of ingestion runs into one scan
```

## Endpoints

| Endpoint                                  | Cost grows with          | Called when                        |
| ----------------------------------------- | ------------------------ | ---------------------------------- |
| `POST /insights/faq/classify`             | the FAQ's structure      | every AI Buddy question            |
| `POST /insights/faq/categories/consolidate` | the number of categories | the category ceiling is crossed    |
| `POST /insights/faq/groups/merge`         | one category's groups    | a group ceiling is crossed         |
| `POST /insights/faq/group`                | every question ever asked | a PM rebuilds the grouping         |
