"""What one unit of a hire's accepted work is called.

Naming that unit in prose -- "celebrate the merges", "shipping is the point" --
is correct for a developer and meaningless for a Scrum Master, a PM or an HR hire,
whose work is just as real and is not a pull request.

The caller supplies these three words and the host renders them into a **fixed**
sentence skeleton. Deliberately three structured fields rather than a blob of
persona text: prose does not compose. Several sources each appending their own
paragraph would produce an incoherent mentor and would blow the context budget
that conversation compaction exists to protect, while three nouns slot into a
skeleton that stays one voice.

The defaults are the engineering wording, so a backend that sends no vocabulary
at all still gets a coherent persona.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Vocabulary:
    """The nouns and verb a hire's accepted work is described with.

    ``contribution_noun`` is bare ("change", "ceremony") because it is always
    rendered next to ``contribution_verb_past``; baking the verb into the noun
    produces "merged merged change" the moment a sentence needs both.
    """

    contribution_noun: str = "change"
    contribution_noun_plural: str = "changes"
    contribution_verb_past: str = "merged"

    def __post_init__(self) -> None:
        noun = self.contribution_noun.strip() if self.contribution_noun else ""
        plural = (
            self.contribution_noun_plural.strip()
            if self.contribution_noun_plural
            else ""
        )
        verb = (
            self.contribution_verb_past.strip() if self.contribution_verb_past else ""
        )
        object.__setattr__(self, "contribution_noun", noun or "change")
        object.__setattr__(self, "contribution_noun_plural", plural or "changes")
        object.__setattr__(self, "contribution_verb_past", verb or "merged")


DEFAULT_VOCABULARY = Vocabulary()
"""Engineering, and the fallback whenever a caller supplies nothing."""
