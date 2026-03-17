"""
Tests for DBLPVerifier.

Unit tests run without the DBLP database (verifier gracefully disabled).
Integration tests require the database to be built first:
    uv run python scripts/build_dblp_index.py

Run unit tests:
    pytest tests/agents/tools/test_dblp.py -v -m "not slow"

Run integration tests (requires built DB):
    pytest tests/agents/tools/test_dblp.py -v -m slow
"""

import pytest
from unittest.mock import patch, MagicMock

from app.models.schemas import ReferenceResult, VerificationSource
from app.agents.tools.verifiers.dblp import (
    DBLPVerifier,
    _normalize_title,
    _build_fts_query,
    _check_author_match,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_ref(
    title: str = "Attention is all you need",
    authors: list[str] | None = None,
    year: int | None = 2017,
) -> ReferenceResult:
    return ReferenceResult(
        title=title,
        authors=authors or ["A. Vaswani", "N. Shazeer"],
        year=year,
        raw_reference=f"{title} ({year})",
    )


# ---------------------------------------------------------------------------
# Unit tests — no database needed
# ---------------------------------------------------------------------------


class TestNormalize:

    def test_lowercases(self):
        assert _normalize_title("BERT: Pre-Training") == "bert pre training"

    def test_strips_punctuation(self):
        result = _normalize_title("PAQ: 65 million questions")
        assert ":" not in result


class TestBuildFtsQuery:

    def test_removes_special_chars(self):
        query = _build_fts_query('BERT: "Pre-training" (deep)')
        assert '"' not in query
        assert "(" not in query
        assert ")" not in query

    def test_limits_to_8_words(self):
        long_title = "word " * 20
        query = _build_fts_query(long_title)
        assert len(query.split()) <= 8

    def test_empty_title_returns_empty(self):
        assert _build_fts_query("") == ""


class TestCheckAuthorMatchDblp:

    def test_match_found(self):
        cited = ["A. Vaswani", "N. Shazeer"]
        dblp = "Ashish Vaswani | Noam Shazeer | Niki Parmar"
        assert _check_author_match(cited, dblp) is True

    def test_no_match(self):
        cited = ["Y. LeCun"]
        dblp = "Ashish Vaswani | Noam Shazeer"
        assert _check_author_match(cited, dblp) is False

    def test_et_al_returns_none(self):
        cited = ["Zirui Guo et al."]
        dblp = "Zirui Guo | Someone Else"
        assert _check_author_match(cited, dblp) is None

    def test_empty_dblp_returns_none(self):
        assert _check_author_match(["A. Vaswani"], "") is None

    def test_none_cited_returns_none(self):
        assert _check_author_match(None, "Ashish Vaswani") is None


class TestDBLPVerifierUnavailable:
    """When DBLP database doesn't exist, verifier should fail gracefully."""

    def setup_method(self):
        self.verifier = DBLPVerifier(db_path="/nonexistent/path/dblp.db")

    def test_available_false_when_db_missing(self):
        assert self.verifier.available is False

    def test_verify_returns_empty_when_unavailable(self):
        ref = make_ref()
        result = self.verifier.verify(ref)
        assert result.sources_checked == []
        assert result.source_results == []

    def test_verify_batch_returns_empty_list(self):
        refs = [make_ref(f"Paper {i}") for i in range(3)]
        results = self.verifier.verify_batch(refs)
        assert len(results) == 3
        assert all(r.sources_checked == [] for r in results)

    def test_search_returns_empty_when_unavailable(self):
        results = self.verifier._search("any title")
        assert results == []


class TestBestMatch:

    def setup_method(self):
        self.verifier = DBLPVerifier(db_path="/nonexistent/dblp.db")

    def test_returns_none_for_empty_candidates(self):
        ref = make_ref()
        assert self.verifier._best_match(ref, []) is None

    def test_returns_none_when_no_title_on_ref(self):
        ref = ReferenceResult(title=None, raw_reference="raw")
        candidates = [{"title": "Some Paper", "authors": "", "year": "2020", "url": ""}]
        assert self.verifier._best_match(ref, candidates) is None

    def test_returns_none_below_threshold(self):
        ref = make_ref(title="Completely Different Title")
        candidates = [
            {
                "title": "Attention Is All You Need",
                "authors": "",
                "year": "2017",
                "url": "",
            }
        ]
        assert self.verifier._best_match(ref, candidates) is None

    def test_returns_best_above_threshold(self):
        ref = make_ref(title="Attention is all you need")
        candidates = [
            {
                "title": "Attention Is All You Need",
                "authors": "Ashish Vaswani | Noam Shazeer",
                "year": "2017",
                "url": "",
            },
        ]
        result = self.verifier._best_match(ref, candidates)
        assert result is not None
        assert result["_title_similarity"] >= 0.85

    def test_prefers_closer_year(self):
        ref = make_ref(title="Attention is all you need", year=2017)
        candidates = [
            {
                "title": "Attention Is All You Need",
                "authors": "",
                "year": "2025",
                "url": "",
                "_title_similarity": 0.95,
            },
            {
                "title": "Attention Is All You Need",
                "authors": "",
                "year": "2017",
                "url": "",
            },
        ]
        # Add similarity score to candidates manually
        for c in candidates:
            if "_title_similarity" not in c:
                c["_title_similarity"] = 0.95
        result = self.verifier._best_match(ref, candidates)
        assert result is not None
        assert result["year"] == "2017"


class TestBuildSourceResult:

    def setup_method(self):
        self.verifier = DBLPVerifier(db_path="/nonexistent/dblp.db")

    def test_not_found(self):
        ref = make_ref()
        result = self.verifier._build_source_result(ref, None)
        assert result.found is False
        assert result.source == VerificationSource.DBLP

    def test_found_populates_fields(self):
        ref = make_ref(title="Attention is all you need", year=2017)
        candidate = {
            "title": "Attention Is All You Need",
            "authors": "Ashish Vaswani | Noam Shazeer",
            "year": "2017",
            "url": "https://arxiv.org/abs/1706.03762",
            "dblp_key": "conf/nips/VaswaniSPUJGKP17",
            "_title_similarity": 0.97,
        }
        result = self.verifier._build_source_result(ref, candidate)
        assert result.found is True
        assert result.title_similarity == 0.97
        assert result.author_match is True
        assert result.year_delta == 0
        assert result.matched_url == "https://arxiv.org/abs/1706.03762"

    def test_falls_back_to_dblp_url_when_no_ee(self):
        ref = make_ref()
        candidate = {
            "title": "Attention Is All You Need",
            "authors": "Ashish Vaswani",
            "year": "2017",
            "url": "",
            "dblp_key": "conf/nips/VaswaniSPUJGKP17",
            "_title_similarity": 0.97,
        }
        result = self.verifier._build_source_result(ref, candidate)
        assert result.matched_url == "https://dblp.org/rec/conf/nips/VaswaniSPUJGKP17"

    def test_year_delta_handles_invalid_year(self):
        ref = make_ref(year=2017)
        candidate = {
            "title": "Attention Is All You Need",
            "authors": "",
            "year": "invalid",
            "url": "",
            "dblp_key": "",
            "_title_similarity": 0.97,
        }
        result = self.verifier._build_source_result(ref, candidate)
        assert result.year_delta is None


# ---------------------------------------------------------------------------
# Integration tests — require built DBLP database
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestDBLPIntegration:
    """
    These tests require the DBLP database to be built first:
        uv run python scripts/build_dblp_index.py
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        from app.core.config import settings

        self.verifier = DBLPVerifier(db_path=settings.dblp_db_path)

        if not self.verifier.available:
            pytest.skip(
                "DBLP database not found. "
                "Build it with: uv run python scripts/build_dblp_index.py"
            )

    def test_attention_paper_found(self):
        ref = ReferenceResult(
            title="Attention is all you need",
            authors=["A. Vaswani", "N. Shazeer"],
            year=2017,
            raw_reference="Vaswani et al. (2017).",
        )
        result = self.verifier.verify(ref)
        sr = result.source_results[0]
        assert sr.found is True
        assert sr.title_similarity >= 0.85

    def test_inpars_acm_sigir_found(self):
        """ACM SIGIR 2022 — no arXiv ID, DBLP should have it."""
        ref = ReferenceResult(
            title="InPars: Unsupervised Dataset Generation for Information Retrieval",
            authors=["Luiz Bonifacio"],
            year=2022,
            raw_reference="Bonifacio et al. (2022). InPars.",
        )
        result = self.verifier.verify(ref)
        sr = result.source_results[0]
        assert sr.found is True
        assert sr.title_similarity >= 0.85

    def test_flair_acl_found(self):
        """FLAIR — ACL Anthology paper, no arXiv, DBLP should have it."""
        ref = ReferenceResult(
            title="FLAIR: An Easy-to-Use Framework for State-of-the-Art NLP",
            authors=["Alan Akbik"],
            year=2019,
            raw_reference="Akbik et al. (2019). FLAIR.",
        )
        result = self.verifier.verify(ref)
        sr = result.source_results[0]
        assert sr.found is True
        assert sr.title_similarity >= 0.85

    def test_fake_paper_not_found(self):
        ref = ReferenceResult(
            title="Quantum Neural Transformers for Distributed Hyperparameter Optimization",
            authors=["A. Fake"],
            year=2023,
            raw_reference="Fake (2023).",
        )
        result = self.verifier.verify(ref)
        assert result.source_results[0].found is False

    def test_batch_sync(self):
        refs = [
            ReferenceResult(
                title="Attention is all you need",
                authors=["A. Vaswani"],
                year=2017,
                raw_reference="Vaswani et al.",
            ),
            ReferenceResult(
                title="Quantum Pasta Optimization",
                authors=["A. Fake"],
                year=2023,
                raw_reference="Fake (2023).",
            ),
        ]
        results = self.verifier.verify_batch(refs)
        assert results[0].source_results[0].found is True
        assert results[1].source_results[0].found is False
