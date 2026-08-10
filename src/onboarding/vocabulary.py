"""What one unit of a hire's accepted work is called.

Onboarding used to end in a merged pull request, so the mentor's persona said so
in prose: "celebrate the merges", "shipping is the point". That is correct for a
developer and meaningless for a Scrum Master, a PM or an HR hire, whose work is
just as real and is not a pull request.

A track supplies these three words and the host renders them into a **fixed**
sentence skeleton. Deliberately three structured fields rather than a blob of
persona text a track could contribute: prose does not compose. Eight tracks each
appending their own paragraph would produce an incoherent mentor and would blow
the context budget that conversation compaction exists to protect, while three
nouns slot into a skeleton that stays one voice no matter how many tracks exist.

The defaults are the engineering wording, so a backend that sends no vocabulary
at all gets exactly the persona it got before tracks existed.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Vocabulary:
    """The nouns and verb one track's accepted work is described with.

    ``contribution_noun`` is bare ("change", "ceremony") because it is always
    rendered next to ``contribution_verb_past``; baking the verb into the noun
    produces "merged merged change" the moment a sentence needs both.
    """

    contribution_noun: str = "change"
    contribution_noun_plural: str = "changes"
    contribution_verb_past: str = "merged"


DEFAULT_VOCABULARY = Vocabulary()
"""Engineering, and the fallback whenever a caller supplies nothing."""
