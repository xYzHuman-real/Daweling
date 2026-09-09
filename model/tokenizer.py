"""A dependency-free byte-level tokenizer for early Daweling training."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DawelingTokenizer:
    """Encode text as UTF-8 bytes with reserved special tokens.

    This is deliberately simple and deterministic. It gives the first model a
    real, owned token interface without depending on a third-party tokenizer.
    A learned subword tokenizer can replace it later without changing the
    model's higher-level contracts.
    """

    pad_id: int = 256
    bos_id: int = 257
    eos_id: int = 258
    unk_id: int = 259

    @property
    def vocab_size(self) -> int:
        return 260

    def encode(self, text: str, *, add_bos: bool = True, add_eos: bool = True) -> list[int]:
        ids = [self.bos_id] if add_bos else []
        ids.extend(text.encode("utf-8"))
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int]) -> str:
        data = bytes(token for token in ids if 0 <= token <= 255)
        return data.decode("utf-8", errors="replace")
