"""Minimal parser for TalkBank CHAT (``.cha``) transcripts used by DementiaBank.

Extracts speaker-attributed utterances, media time bullets (``\\x15start_end\\x15`` in milliseconds) and
``@ID`` header metadata. CHAT annotation codes are stripped to recover plain words; filled pauses written
as ``&uh`` / ``&um`` are kept as plain fillers because they are linguistically informative.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

_BULLET = re.compile(r"\x15(\d+)_(\d+)\x15")
_SPEAKER_LINE = re.compile(r"^\*([A-Z]{3}):\s*(.*)$")


@dataclass
class ChatUtterance:
    speaker: str
    text: str
    start: float | None
    end: float | None


@dataclass
class ChatTranscript:
    utterances: list[ChatUtterance] = field(default_factory=list)
    participants: dict[str, dict[str, str]] = field(default_factory=dict)
    media: str | None = None

    def speaker_text(self, speaker: str = "PAR") -> str:
        return " ".join(u.text for u in self.utterances if u.speaker == speaker and u.text).strip()

    def turns(self) -> list[dict]:
        """Speaker turns with timing; only utterances with time bullets are returned."""
        return [
            {"speaker": u.speaker, "start": u.start, "end": u.end, "text": u.text}
            for u in self.utterances
            if u.start is not None and u.end is not None
        ]


def clean_chat_text(raw: str) -> str:
    text = _BULLET.sub(" ", raw)
    text = re.sub(r"\[[^\]]*\]", " ", text)  # [//] [/] [+ exc] [: target] error codes
    text = re.sub(r"&=\S+", " ", text)  # &=laughs  (non-verbal events)
    text = re.sub(r"&([a-z]+)", r"\1", text)  # &uh -> uh (filled pauses / fragments)
    text = re.sub(r"\(\.+\)", " ", text)  # (.) (..) pauses
    text = re.sub(r"\b(xxx|yyy|www)\b", " ", text)  # unintelligible
    text = re.sub(r"[<>‡„+\"/^~@]|\b0\w+", " ", text)
    text = re.sub(r"\(([a-z]+)\)", r"\1", text)  # (be)cause -> because
    text = re.sub(r"[_:]", " ", text)
    text = re.sub(r"\s+([.?!,])", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_chat(path: str | Path) -> ChatTranscript:
    transcript = ChatTranscript()
    current: list[str] | None = None
    speaker = ""
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()

    def flush() -> None:
        if current is None:
            return
        raw = " ".join(current)
        bullets = _BULLET.findall(raw)
        start = int(bullets[0][0]) / 1000 if bullets else None
        end = int(bullets[-1][1]) / 1000 if bullets else None
        transcript.utterances.append(ChatUtterance(speaker, clean_chat_text(raw), start, end))

    for line in lines:
        if line.startswith("@ID:"):
            fields = line.split(":", 1)[1].strip().split("|")
            if len(fields) >= 8:
                transcript.participants[fields[2]] = {
                    "language": fields[0], "corpus": fields[1], "age": fields[3].rstrip(";"),
                    "sex": fields[4], "group": fields[5], "role": fields[7],
                    "education": fields[8] if len(fields) > 8 else "",
                }
        elif line.startswith("@Media:"):
            transcript.media = line.split(":", 1)[1].split(",")[0].strip()
        match = _SPEAKER_LINE.match(line)
        if match:
            flush()
            speaker, current = match.group(1), [match.group(2)]
        elif line.startswith("\t") and current is not None:
            current.append(line.strip())
        elif line.startswith(("%", "@")):
            flush()
            current = None
    flush()
    return transcript
