from dataclasses import dataclass


@dataclass(frozen=True)
class TextRecord:
    text: str
    label: int
    domain: str
    split: str
