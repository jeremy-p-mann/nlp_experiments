import pytest

from src import Corpus, get_corpus


@pytest.fixture(scope='module')
def btrel_corpus() -> Corpus:
    return get_corpus('btrel')


@pytest.fixture(scope='module')
def random_seed() -> int:
    return 42


@pytest.fixture(scope='module')
def n_samples() -> int:
    return 10


@pytest.fixture(scope='module')
def docs(btrel_corpus, random_seed, n_samples):
    docs = btrel_corpus.sample(N=10, random_seed=42)
    return docs


@pytest.fixture(scope='module')
def docs_series(btrel_corpus, random_seed, n_samples):
    docs = btrel_corpus.sample(N=10, random_seed=42)
    return docs


def test_btrel_corpus_contains_sample_documents(docs, n_samples):
    assert len(docs) == n_samples
