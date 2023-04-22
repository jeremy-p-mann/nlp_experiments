from enum import Enum
from typing import Optional, Protocol, runtime_checkable

from src.btrel import get_btrel_corpus


@runtime_checkable
class Docs(Protocol):
    def __len__(self,) -> int:
        ...

    def __getitem__(self, id: int) -> Optional[str]:
        ...


@runtime_checkable
class Corpus(Protocol):
    def sample(self, N: int, random_seed: int = 42, **kwargs) -> Docs:
        ...


def get_corpus(corpus: str) -> Corpus:
    return get_btrel_corpus()
