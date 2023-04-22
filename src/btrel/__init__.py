from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class Bigrams:
    __values: pd.Dataframe

    def __len__(self,) -> int:
        return len(self.__values)

    @property
    def context_words(self,) -> pd.Series:
        return self.__values.loc['context']

    @property
    def focal_words(self,) -> pd.Series:
        return self.__values['focal']


@dataclass
class BTRELDocs:
    __values: pd.Series

    def __len__(self,) -> int:
        return len(self.__values)

    def sample_bigrams(self,) -> Bigrams:
        assert False


@dataclass
class BTRELCorpus:
    __values: pd.Series

    def sample(self, N: int, random_seed: int = 42, **kwargs) -> BTRELDocs:
        return BTRELDocs(self.__values.sample(n=N, random_state=42, replace=True))


def get_btrel_corpus() -> BTRELCorpus:
    values = pd.Series([
        'I go home',
        'I go outside',
        'he goes outside',
        'he goes home',
    ])
    return BTRELCorpus(values)
