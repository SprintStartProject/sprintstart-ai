"""Short descriptive titles generated from a prompt."""

from llm.base import LLMClient, Message

SYSTEM_PROMPT = """
You generate short descriptive titles.

Rules:

- return only the title
- no quotes
- no explanations
- respect the character limit
"""


def generate_title(llm: LLMClient, prompt: str, max_length: int) -> str:
    """A title for ``prompt`` of at most ``max_length`` characters.

    Whitespace around the model's answer is dropped, and an answer that is still
    too long is cut hard at the limit. ``LLMUnavailableError`` propagates.
    """
    messages: list[Message] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Create a title with maximum {max_length} characters.\n\n{prompt}"
            ),
        },
    ]
    title = llm.generate(messages=messages).strip()
    if len(title) > max_length:
        title = title[:max_length].rstrip()
    return title
