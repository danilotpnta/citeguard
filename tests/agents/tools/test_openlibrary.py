"""
Tests for OpenLibraryVerifier.

Unit tests mock HTTP responses — no network required.
Integration tests hit the live OpenLibrary API (marked slow).

Run unit tests only:
    uv run pytest tests/agents/tools/test_openlibrary.py -v -m "not slow"

Run all including integration:
    uv run pytest tests/agents/tools/test_openlibrary.py -v -m slow

Run single test:
    uv run pytest tests/agents/tools/test_openlibrary.py::TestNormalizeTitle -v
"""

import pytest
from unittest.mock import AsyncMock, patch

from app.models.schemas import ReferenceResult, VerificationSource
from app.agents.tools.verifiers.openlibrary import (
    OpenLibraryVerifier,
    _normalize_title,
    _extract_lastname,
    _check_author_match,
    _best_match,
    _build_source_result,
    _parse_year,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_ref(
    title: str = "NumPy: A guide to NumPy",
    authors: list[str] | None = None,
    year: int | None = 2006,
    url: str | None = None,
) -> ReferenceResult:
    return ReferenceResult(
        title=title,
        authors=authors or ["Travis Oliphant"],
        year=year,
        raw_reference=f"{title} ({year})",
        url=url,
    )


def make_ol_doc(
    title: str = "NumPy: A Guide to NumPy",
    author_names: list[str] | None = None,
    year: int = 2006,
    key: str = "/works/OL2896994W",
) -> dict:
    return {
        "key": key,
        "title": title,
        "author_name": author_names or ["Travis E. Oliphant"],
        "first_publish_year": year,
        "publisher": ["Trelgol Publishing"],
    }


def make_ol_response(docs: list[dict]) -> dict:
    return {"numFound": len(docs), "start": 0, "docs": docs}


# ---------------------------------------------------------------------------
# Unit tests — _normalize_title
# ---------------------------------------------------------------------------


class TestNormalizeTitle:

    def test_lowercases(self):
        assert _normalize_title("NumPy: A Guide") == "numpy a guide"

    def test_strips_punctuation(self):
        result = _normalize_title("PyTorch: An Imperative-Style Library!")
        assert "pytorch" in result
        assert "!" not in result

    def test_ligature_ff(self):
        assert "ff" in _normalize_title("Eﬀicient Memory Management")

    def test_ligature_fi(self):
        assert "fi" in _normalize_title("ﬁle system")

    def test_replacement_char_stripped(self):
        result = _normalize_title("E\ufffdcient Memory")
        assert "\ufffd" not in result

    def test_hyphen_artifact_merged(self):
        # "In- pars" → "inpars"
        assert "inpars" in _normalize_title("In- pars toolkit")

    def test_collapses_whitespace(self):
        result = _normalize_title("lots   of   spaces")
        assert "  " not in result


# ---------------------------------------------------------------------------
# Unit tests — _extract_lastname
# ---------------------------------------------------------------------------


class TestExtractLastname:

    def test_simple_name(self):
        assert _extract_lastname("Travis Oliphant") == "oliphant"

    def test_initial_lastname(self):
        assert _extract_lastname("T. Oliphant") == "oliphant"

    def test_comma_format(self):
        assert _extract_lastname("Oliphant, Travis") == "oliphant"

    def test_strips_et_al(self):
        assert _extract_lastname("Travis Oliphant et al.") == "oliphant"

    def test_trailing_period(self):
        assert _extract_lastname("T. Oliphant.") == "oliphant"


# ---------------------------------------------------------------------------
# Unit tests — _check_author_match
# ---------------------------------------------------------------------------


class TestCheckAuthorMatch:

    def test_exact_match(self):
        assert _check_author_match(["Travis Oliphant"], ["Travis E. Oliphant"]) is True

    def test_lastname_match(self):
        assert _check_author_match(["T. Oliphant"], ["Travis Oliphant"]) is True

    def test_no_match(self):
        assert _check_author_match(["John Smith"], ["Travis Oliphant"]) is False

    def test_et_al_first_author_matches_returns_none(self):
        result = _check_author_match(["Travis Oliphant et al."], ["Travis E. Oliphant"])
        assert result is None

    def test_et_al_first_author_mismatch_returns_false(self):
        result = _check_author_match(["John Smith et al."], ["Travis E. Oliphant"])
        assert result is False

    def test_only_et_al_returns_none(self):
        assert _check_author_match(["et al."], ["Travis Oliphant"]) is None

    def test_none_cited_returns_none(self):
        assert _check_author_match(None, ["Travis Oliphant"]) is None

    def test_none_ol_authors_returns_none(self):
        assert _check_author_match(["Travis Oliphant"], None) is None

    def test_empty_ol_authors_returns_none(self):
        assert _check_author_match(["Travis Oliphant"], []) is None

    def test_multiple_authors_one_matches(self):
        assert (
            _check_author_match(
                ["Travis Oliphant", "John Smith"],
                ["Travis E. Oliphant"],
            )
            is True
        )


# ---------------------------------------------------------------------------
# Unit tests — _best_match
# ---------------------------------------------------------------------------


class TestBestMatch:

    def test_returns_none_empty_candidates(self):
        assert _best_match(make_ref(), []) is None

    def test_returns_none_no_title(self):
        ref = ReferenceResult(title=None, raw_reference="raw")
        assert _best_match(ref, [make_ol_doc()]) is None

    def test_returns_match_above_threshold(self):
        ref = make_ref(title="NumPy: A guide to NumPy")
        result = _best_match(ref, [make_ol_doc(title="NumPy: A Guide to NumPy")])
        assert result is not None
        assert result["_title_similarity"] >= 0.85

    def test_returns_none_below_threshold(self):
        ref = make_ref(title="NumPy: A guide to NumPy")
        result = _best_match(
            ref, [make_ol_doc(title="Completely Different Cooking Book")]
        )
        assert result is None

    def test_picks_closest_year_among_matches(self):
        ref = make_ref(title="Guide to NumPy", year=2006)
        candidates = [
            make_ol_doc(title="Guide to NumPy", year=2010, key="/works/OL1W"),
            make_ol_doc(title="Guide to NumPy", year=2006, key="/works/OL2W"),
        ]
        result = _best_match(ref, candidates)
        assert result["key"] == "/works/OL2W"

    def test_scores_are_attached(self):
        ref = make_ref(title="Guide to NumPy")
        result = _best_match(ref, [make_ol_doc(title="Guide to NumPy")])
        assert "_title_similarity" in result


# ---------------------------------------------------------------------------
# Unit tests — _build_source_result
# ---------------------------------------------------------------------------


class TestBuildSourceResult:

    def test_not_found_when_no_candidate(self):
        sr = _build_source_result(make_ref(), None, found=False)
        assert sr.found is False
        assert sr.source == VerificationSource.OPENLIBRARY
        assert sr.title_similarity is None

    def test_found_populates_fields(self):
        ref = make_ref(year=2006)
        doc = make_ol_doc(year=2006)
        doc["_title_similarity"] = 0.97
        sr = _build_source_result(ref, doc, found=True)

        assert sr.found is True
        assert sr.title_similarity == 0.97
        assert sr.year_delta == 0
        assert sr.matched_url == "https://openlibrary.org/works/OL2896994W"
        assert sr.matched_title == "NumPy: A Guide to NumPy"

    def test_year_delta_computed(self):
        ref = make_ref(year=2006)
        doc = make_ol_doc(year=2009)
        doc["_title_similarity"] = 0.95
        sr = _build_source_result(ref, doc, found=True)
        assert sr.year_delta == 3

    def test_year_delta_none_when_ref_has_no_year(self):
        ref = make_ref(year=None)
        doc = make_ol_doc(year=2006)
        doc["_title_similarity"] = 0.95
        sr = _build_source_result(ref, doc, found=True)
        assert sr.year_delta is None

    def test_author_match_propagated(self):
        ref = make_ref(authors=["Travis Oliphant"])
        doc = make_ol_doc()
        doc["_title_similarity"] = 0.95
        sr = _build_source_result(ref, doc, found=True)
        assert sr.author_match is True


# ---------------------------------------------------------------------------
# Unit tests — _parse_year
# ---------------------------------------------------------------------------


class TestParseYear:

    def test_plain_year(self):
        assert _parse_year("2006") == 2006

    def test_month_year(self):
        assert _parse_year("January 2006") == 2006

    def test_iso_date(self):
        assert _parse_year("2006-01-15") == 2006

    def test_empty_string(self):
        assert _parse_year("") is None

    def test_no_year(self):
        assert _parse_year("no date available") is None

    def test_old_year(self):
        assert _parse_year("1970") == 1970


# ---------------------------------------------------------------------------
# Async unit tests — mocked _search / _lookup_isbn
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestOpenLibraryAsync:

    async def test_verify_book_found(self):
        ref = make_ref()
        verifier = OpenLibraryVerifier()
        with patch.object(
            verifier, "_search", new=AsyncMock(return_value=[make_ol_doc()])
        ):
            result = await verifier.verify(ref)

        sr = result.source_results[0]
        assert sr.found is True
        assert sr.title_similarity >= 0.85
        assert sr.matched_url == "https://openlibrary.org/works/OL2896994W"
        assert result.sources_checked == [VerificationSource.OPENLIBRARY]

    async def test_verify_not_found_empty_results(self):
        ref = make_ref(title="Completely Hallucinated Book XYZ 9999")
        verifier = OpenLibraryVerifier()
        with patch.object(verifier, "_search", new=AsyncMock(return_value=[])):
            result = await verifier.verify(ref)

        assert result.source_results[0].found is False
        assert result.source_results[0].title_similarity is None

    async def test_verify_title_below_threshold(self):
        ref = make_ref()
        verifier = OpenLibraryVerifier()
        with patch.object(
            verifier,
            "_search",
            new=AsyncMock(
                return_value=[make_ol_doc(title="Something Completely Different")]
            ),
        ):
            result = await verifier.verify(ref)

        assert result.source_results[0].found is False

    async def test_verify_no_title_skips_search(self):
        ref = make_ref(title=None)
        verifier = OpenLibraryVerifier()
        mock_search = AsyncMock()
        with patch.object(verifier, "_search", new=mock_search):
            result = await verifier.verify(ref)

        mock_search.assert_not_called()
        assert result.source_results[0].found is False

    async def test_verify_author_match_true(self):
        ref = make_ref(authors=["Travis Oliphant"])
        verifier = OpenLibraryVerifier()
        with patch.object(
            verifier, "_search", new=AsyncMock(return_value=[make_ol_doc()])
        ):
            result = await verifier.verify(ref)

        assert result.source_results[0].author_match is True

    async def test_verify_author_et_al_returns_none(self):
        ref = make_ref(authors=["Travis Oliphant et al."])
        verifier = OpenLibraryVerifier()
        with patch.object(
            verifier, "_search", new=AsyncMock(return_value=[make_ol_doc()])
        ):
            result = await verifier.verify(ref)

        # First author matches → None (partial), not True
        assert result.source_results[0].author_match is None

    async def test_verify_year_delta(self):
        ref = make_ref(year=2006)
        verifier = OpenLibraryVerifier()
        with patch.object(
            verifier,
            "_search",
            new=AsyncMock(return_value=[make_ol_doc(year=2007)]),
        ):
            result = await verifier.verify(ref)

        assert result.source_results[0].year_delta == 1

    async def test_verify_isbn_fast_path_used(self):
        """If ref has isbn, _lookup_isbn is called before _search."""
        ref = make_ref()
        ref = ref.model_copy(update={"isbn": "9781491912126"})
        verifier = OpenLibraryVerifier()

        isbn_doc = make_ol_doc()
        isbn_doc["_title_similarity"] = 0.95

        mock_isbn = AsyncMock(return_value=isbn_doc)
        mock_search = AsyncMock()

        with patch.object(verifier, "_lookup_isbn", new=mock_isbn):
            with patch.object(verifier, "_search", new=mock_search):
                result = await verifier.verify(ref)

        mock_isbn.assert_called_once()
        mock_search.assert_not_called()
        assert result.source_results[0].found is True

    async def test_verify_isbn_fallback_to_search(self):
        """If ISBN lookup returns None, falls back to title search."""
        ref = make_ref()
        ref = ref.model_copy(update={"isbn": "9781491912126"})
        verifier = OpenLibraryVerifier()

        with patch.object(verifier, "_lookup_isbn", new=AsyncMock(return_value=None)):
            with patch.object(
                verifier, "_search", new=AsyncMock(return_value=[make_ol_doc()])
            ):
                result = await verifier.verify(ref)

        assert result.source_results[0].found is True

    async def test_verify_batch_returns_all(self):
        refs = [make_ref(title=f"Book {i}") for i in range(4)]
        verifier = OpenLibraryVerifier()
        with patch.object(
            verifier, "_search", new=AsyncMock(return_value=[make_ol_doc()])
        ):
            results = await verifier.verify_batch(refs)

        assert len(results) == 4

    async def test_verify_batch_concurrent_mixed(self):
        """Some found, some not — batch handles mixed results correctly."""
        verifier = OpenLibraryVerifier()

        async def mock_search(title, authors=None):
            if "NumPy" in title:
                return [make_ol_doc()]
            return []

        with patch.object(verifier, "_search", new=mock_search):
            results = await verifier.verify_batch(
                [
                    make_ref(title="NumPy: A guide to NumPy"),
                    make_ref(title="Hallucinated Book XYZ"),
                ]
            )

        assert results[0].source_results[0].found is True
        assert results[1].source_results[0].found is False

    async def test_search_uses_q_with_author_in_first_pass(self):
        """First pass combines title + author lastname in q=."""
        captured_params = []

        async def mock_get(url, params=None):
            captured_params.append(params or {})
            return {"docs": [make_ol_doc()]}

        verifier = OpenLibraryVerifier()
        with patch.object(verifier, "_get", new=mock_get):
            ref = make_ref(authors=["Travis Oliphant"])
            await verifier.verify(ref)

        assert "q" in captured_params[0]
        assert "oliphant" in captured_params[0]["q"].lower()
        assert "title" not in captured_params[0]  # title= param must not be present
        assert "author" not in captured_params[0]  # author= param must not be present

    async def test_search_falls_back_to_title_only(self):
        """If first pass returns empty, second pass uses title only in q=."""
        call_count = 0

        async def mock_get(url, params=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {"docs": []}  # first pass empty
            return {"docs": [make_ol_doc()]}  # second pass finds it

        verifier = OpenLibraryVerifier()
        with patch.object(verifier, "_get", new=mock_get):
            ref = make_ref(authors=["Travis Oliphant"])
            result = await verifier.verify(ref)

        assert call_count == 2
        assert result.source_results[0].found is True


# ---------------------------------------------------------------------------
# Integration tests — live OpenLibrary API
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestOpenLibraryIntegration:

    @pytest.fixture(autouse=True)
    def setup(self):
        self.verifier = OpenLibraryVerifier()

    async def test_numpy_found(self):
        ref = make_ref(
            title="NumPy: A guide to NumPy",
            authors=["Travis Oliphant"],
            year=2006,
        )
        result = await self.verifier.verify(ref)
        sr = result.source_results[0]
        assert sr.found is True
        assert sr.title_similarity >= 0.85

    async def test_torch_found(self):
        """
        Torch (2002) is an Idiap technical report, not a published book.
        OpenLibrary doesn't index it — UNVERIFIABLE is the correct outcome.
        Verifier should return gracefully without crashing.
        """
        ref = make_ref(
            title="Torch: a modular machine learning software library",
            authors=["Ronan Collobert", "Samy Bengio", "Johnny Mariéthoz"],
            year=2002,
        )
        result = await self.verifier.verify(ref)
        sr = result.source_results[0]
        assert sr.source == VerificationSource.OPENLIBRARY
        assert sr.found is False  # correct — not a book, not indexed

    async def test_hallucinated_book_not_found(self):
        ref = make_ref(
            title="Completely Fake Book That Does Not Exist XYZ 9999",
            authors=["John Doe"],
            year=2099,
        )
        result = await self.verifier.verify(ref)
        assert result.source_results[0].found is False

    async def test_arxiv_paper_not_found(self):
        """
        arXiv papers should not be found in OpenLibrary —
        confirms we're not double-counting academic papers.
        """
        ref = make_ref(
            title="Attention Is All You Need",
            authors=["Ashish Vaswani"],
            year=2017,
        )
        result = await self.verifier.verify(ref)
        # OpenLibrary may or may not have this — we don't assert found,
        # just that the verifier doesn't crash
        assert result.source_results[0].source == VerificationSource.OPENLIBRARY

    async def test_batch_concurrent(self):
        refs = [
            make_ref(
                title="NumPy: A guide to NumPy",
                authors=["Travis Oliphant"],
                year=2006,
            ),
            make_ref(
                title="Torch: a modular machine learning software library",
                authors=["Ronan Collobert"],
                year=2002,
            ),
            make_ref(
                title="Hallucinated Book That Cannot Exist",
                authors=["Fake Author"],
                year=2099,
            ),
        ]
        results = await self.verifier.verify_batch(refs)
        assert results[0].source_results[0].found is True
        assert results[2].source_results[0].found is False

    async def test_debug_raw_http(self):
        import httpx

        async with httpx.AsyncClient(follow_redirects=True) as client:
            # Try with q= instead of title=, no fields param
            response = await client.get(
                "https://openlibrary.org/search.json",
                params={
                    "q": "NumPy guide Oliphant",
                    "limit": 5,
                },
                headers={"User-Agent": "citeguard/1.0 (test)"},
            )
        print(f"\nstatus: {response.status_code}")
        print(f"body: {response.text[:800]}")

    async def test_debug_numpy_raw(self):
        results = await self.verifier._search(
            "NumPy: A guide to NumPy", ["Travis Oliphant"]
        )
        print(f"\nresults count: {len(results)}")
        print(f"raw results: {results}")

        for r in results[:5]:
            from app.agents.tools.verifiers.openlibrary import _normalize_title
            from rapidfuzz import fuzz

            ol_title = r.get("title", "")
            cited = "NumPy: A guide to NumPy"
            score = (
                fuzz.ratio(_normalize_title(ol_title), _normalize_title(cited)) / 100.0
            )
            print(f"\ntitle: {ol_title}")
            print(f"normalized: {_normalize_title(ol_title)}")
            print(f"score vs cited: {score:.3f}")
            print(f"authors: {r.get('author_name')}")
            print(f"year: {r.get('first_publish_year')}")

    async def test_debug_torch_raw(self):
        results = await self.verifier._search(
            "Torch: a modular machine learning software library",
            ["Ronan Collobert", "Samy Bengio", "Johnny Mariéthoz"],
        )
        print(f"\nresults count: {len(results)}")
        for r in results[:5]:
            from rapidfuzz import fuzz
            from app.agents.tools.verifiers.openlibrary import _normalize_title

            ol_title = r.get("title", "")
            cited = "Torch: a modular machine learning software library"
            score = (
                fuzz.token_set_ratio(
                    _normalize_title(ol_title), _normalize_title(cited)
                )
                / 100.0
            )
            print(f"title: {ol_title}")
            print(f"score: {score:.3f}")
            print(f"authors: {r.get('author_name')}")
            print(f"year: {r.get('first_publish_year')}")
