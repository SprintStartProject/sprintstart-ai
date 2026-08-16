# FAQ grouping & title concept

Design record for stories #284 and #285 (epic #62, Analytics/Evaluation/Experimentation), from the
Sprint 6 review on 2026-08-12. It describes how the FAQ insight is structured, how it stays
current, and how it is kept readable as the question set grows. It spans three repositories, so it
lives here, where the classification itself does.

## The problem

The original FAQ was a **flat list of question groups, rebuilt on demand**. A PM pressed refresh,
the backend sent every question the project had ever asked to the AI service, one LLM pass
clustered all of them, and the result replaced the previous one. Each group was headlined by one of
its member questions, verbatim.

Three things follow from that shape, and all were raised in the review:

1. **It cannot be current.** The rebuild's cost grows with the total number of questions, so it can
   never run per chat message. The panel is stale by default, and a PM has no way of knowing by how
   much.
2. **It cannot scale.** Recurring questions accumulate forever, and the list only grows.
3. **It is slow to read.** A verbatim question as a headline makes a PM read a whole sentence per
   entry to work out what it is about.

## The structure

**One level.** An entry is one recurring question — the same thing asked in different words —
carrying a short generated **title**, a count, its answering documents, and the individual
phrasings underneath it.

An earlier iteration of this design had two levels, with topic categories above the groups. It was
built and then removed: on screen it produced sections containing sections, and the extra level did
not earn its place — the thing that actually made it readable was the *summarising title*, not the
nesting. So the title moved down to the entry itself and the category level went away.

That is the answer to #285's "grouping concept that scales": what makes the list scannable is that
each entry says what it is about in three to eight words ("Getting VPN access") instead of
requiring a sentence to be read, plus the ceiling and ordering below. Not another level of
hierarchy.

A title is generated, never copied verbatim from a question, and must stay specific enough to tell
neighbouring entries apart — "Setup" alone is useless next to five other setup questions. When the
model gives nothing usable, the entry falls back to its redacted representative question as the
title: wordy, but it still says what the entry is about, unlike an "Untitled" placeholder.

Titles are **stable**. A matched entry keeps its own title rather than taking whatever the
classifier proposed for the incoming phrasing — otherwise the list would churn every time someone
rephrases a question.

## Staying current (#284)

The rebuild is no longer the normal path. It is the fallback.

```
user asks a question in the chat
  → chat stores the message, publishes ChatQuestionAskedEvent (async, fire-and-forget)
  → insights calls POST /insights/faq/classify with: the question
                                                   + candidate entries (≤ candidate limit)
  → AI answers: relevant? · existing entry or a new one · its title · redacted text
  → insights applies it, then enforces the entry ceiling
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
- **Retrieval only runs for a new entry.** An existing one already carries the documents that
  answer it.
- **Redaction is folded into the same call**, rather than a second round-trip per message.
- **Candidates carry both title and verbatim question.** A summarised title can lose the component
  name that distinguishes "start the frontend" from "start the backend", so the wording travels
  with it.

### Why not exact

One question carries little context. Judged against a bounded candidate list, the classifier will
occasionally open an entry that duplicates an existing one — most likely one created minutes ago,
which has a count of one and would not survive a purely frequency-ranked candidate cut. The
candidate selection mitigates this (half the budget goes to the most-asked entries, half to the
most recently asked), but does not eliminate it.

The alternative — comparing every question against the full corpus — is exactly the cost this
design exists to avoid. So the drift is accepted and then **bounded**, by the merge pass below. The
system is self-healing rather than exact.

## Staying readable (#285)

One ceiling, configurable under `sprintstart.insights.faq`:

| Ceiling      | Default | Crossed → |
| ------------ | ------- | --------- |
| `max-groups` | 40      | `POST /insights/faq/groups/merge` proposes merges of duplicate entries |

It is enforced by **folding entries together, never by refusing one** — a limit must not lose a
question.

The merge pass is cheap for the same reason the classification is: it sends structure, not history.
It sees each entry's id, title, representative question and count — never the phrasings stored
underneath. That is why it can run on crossing the ceiling instead of being scheduled nightly.

**The plan is a proposal.** The AI service validates its own output (unknown ids, a source claimed
twice, a target that is itself merged away), and the backend validates it again before applying it,
because the backend is the one applying it destructively. A merge that cannot be applied safely is
dropped, not guessed at. An empty plan is a legitimate answer: staying over the limit beats merging
two distinct questions into one.

The surviving entry is always one that already exists — it keeps the stored samples, title and
documents — and absorbs the other's count, timestamps, phrasings and any documents it did not
already cite.

## Growing and shifting over time (#285)

A count alone cannot distinguish a topic that is picking up from one that was asked constantly a
year ago and never since — but to a PM deciding where documentation effort pays off, those mean
opposite things.

So every entry also carries a **trend**: questions in the current window (default 14 days) against
the window before it.

- `RISING` — asked more than before
- `STEADY` — asked about as often
- `FADING` — asked less, **or not at all in either window**

The last clause is the one that matters. Two empty windows are not a topic holding its level; they
are a topic nobody has asked about in a month.

Trends are measured over the questions actually stored for an entry. Every question arriving
through the live path is stored, so the numbers are exact for a FAQ maintained that way. Directly
after a full rebuild — which carries back only a redacted sample per entry — they understate large
entries until live traffic accumulates again. The entry's *recency* (`firstAskedAt` /
`lastAskedAt`) stays exact either way, because the rebuild returns the ids of every question it
grouped and the backend maps those back to the messages they were asked in.

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

The rebuild titles its entries too, so it cannot undo the titling it is meant to repair.

## Configuration

```yaml
sprintstart:
  insights:
    faq:
      live-updates: true      # file questions as they are asked
      max-groups: 40          # ceiling before duplicate entries are merged
      candidate-groups: 40    # entries one classification may consider
      sample-questions: 10    # phrasings shown on an entry's detail view
      trend-window-days: 14   # window a trend is measured over
    knowledge-gaps:
      auto-refresh: true      # rescan once new documentation is indexed
      debounce-seconds: 60    # coalesce a burst of ingestion runs into one scan
```

## Endpoints

| Endpoint                          | Cost grows with           | Called when                |
| --------------------------------- | ------------------------- | -------------------------- |
| `POST /insights/faq/classify`     | the candidate list        | every chat question        |
| `POST /insights/faq/groups/merge` | the number of entries     | the entry ceiling is crossed |
| `POST /insights/faq/group`        | every question ever asked | a PM rebuilds the grouping |
