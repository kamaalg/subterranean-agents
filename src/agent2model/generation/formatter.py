"""Convert generated conversations into HF chat-template JSONL.

The training data the fine-tuner consumes is plain dialogue — alternating user
and assistant turns — with **no trace of the flowchart**. Node ids, decision
branches, ``when`` conditions, and scenario variables must never appear in the
output; the compiled model only ever sees natural conversation.

Each conversation becomes one JSONL line::

    {"messages": [{"role": "user", "content": "..."},
                  {"role": "assistant", "content": "..."}]}

``user``-role flowchart turns map to ``role: "user"`` and ``agent``-role turns
map to ``role: "assistant"`` (the model being trained plays the agent).
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

ChatRole = Literal["user", "assistant"]
"""Role in the HF chat-template output."""


class Turn(BaseModel):
    """One generated turn of dialogue.

    Attributes:
        role: ``"user"`` (the simulated customer) or ``"assistant"`` (the agent
            being compiled).
        content: The natural-language text of the turn.
    """

    model_config = ConfigDict(extra="forbid")

    role: ChatRole
    content: str


class Conversation(BaseModel):
    """A complete generated conversation, ready to be formatted.

    Attributes:
        turns: Ordered dialogue turns. Only ``role``/``content`` survive into the
            dataset — any flowchart structure used to *produce* the turns is
            deliberately not stored here.
    """

    model_config = ConfigDict(extra="forbid")

    turns: list[Turn]

    def to_chat_messages(self) -> dict[str, list[dict[str, str]]]:
        """Render this conversation as a single HF chat-template record.

        Returns:
            A ``{"messages": [...]}`` mapping with one entry per turn.

        Example:
            >>> Conversation(turns=[Turn(role="user", content="hi")]).to_chat_messages()
            {'messages': [{'role': 'user', 'content': 'hi'}]}
        """
        return {"messages": [{"role": t.role, "content": t.content} for t in self.turns]}


def write_dataset(conversations: Iterable[Conversation], path: str | Path) -> int:
    """Write conversations to a JSONL dataset in HF chat-template format.

    The parent directory is created if needed. One conversation is written per
    line; empty conversations (no turns) are skipped so the dataset never
    contains blank records.

    Args:
        conversations: The conversations to serialise.
        path: Destination ``.jsonl`` file (typically ``build/<name>/dataset.jsonl``).

    Returns:
        The number of conversations written.

    Example:
        >>> n = write_dataset([Conversation(turns=[Turn(role="user", content="hi")])], "ds.jsonl")
        >>> n
        1
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out.open("w", encoding="utf-8") as fh:
        for conv in conversations:
            if not conv.turns:
                continue
            fh.write(json.dumps(conv.to_chat_messages(), ensure_ascii=False))
            fh.write("\n")
            written += 1
    return written
