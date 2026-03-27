"""
Tests for merge_results_node and score_node.

Pure logic — no API calls, no LLM calls, no network.
All tests run without mocking anything external.

Run with:
    pytest tests/test_merge_and_score.py -v
"""

import pytest
from app.models.schemas import (
    ReferenceResult,
    VerificationResult,
    SourceResult,
    VerificationSource,
)
from app.graph.nodes.merge_and_score_nodes import (
    merge_results_node,
    score_node,
    _determine_verdict,
    _looks_non_academic,
    VERIFIED,
    LIKELY_REAL,
    NEEDS_REVIEW,
    UNVERIFIABLE,
    LIKELY_HALLUCINATED,
    RETRACTED,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_ref(
    title: str = "Some Paper",
    authors: list[str] | None = None,
    year: int | None = 2020,
    doi: str | None = None,
    arxiv_id: str | None = None,
    url: str | None = None,
    venue: str | None = None,
) -> ReferenceResult:
    return ReferenceResult(
        title=title,
        authors=authors or ["A. Author"],
        year=year,
        doi=doi,
        arxiv_id=arxiv_id,
        url=url,
        venue=venue,
        raw_reference=f"{title} raw",
    )


def make_source_result(
    source: VerificationSource = VerificationSource.CROSSREF,
    found: bool = True,
    title_similarity: float | None = 0.95,
    author_match: bool | None = True,
    retracted: bool = False,
) -> SourceResult:
    return SourceResult(
        source=source,
        found=found,
        title_similarity=title_similarity,
        author_match=author_match,
        retracted=retracted,
    )


def make_verification_result(
    ref: ReferenceResult,
    source: VerificationSource = VerificationSource.CROSSREF,
    found: bool = True,
    title_similarity: float | None = 0.95,
    author_match: bool | None = True,
    retracted: bool = False,
) -> VerificationResult:
    return VerificationResult(
        reference=ref,
        sources_checked=[source],
        source_results=[
            make_source_result(source, found, title_similarity, author_match, retracted)
        ],
    )


# ---------------------------------------------------------------------------
# _looks_non_academic
# ---------------------------------------------------------------------------

class TestLooksNonAcademic:

    def test_github_url_is_non_academic(self):
        ref = make_ref(url="https://github.com/some/repo")
        assert _looks_non_academic(ref) is True

    def test_openai_blog_is_non_academic(self):
        ref = make_ref(url="https://openai.com/blog/gpt4")
        assert _looks_non_academic(ref) is True

    def test_mit_press_venue_is_non_academic(self):
        ref = make_ref(venue="MIT Press")
        assert _looks_non_academic(ref) is True

    def test_journal_paper_is_academic(self):
        ref = make_ref(venue="Nature", url=None)
        assert _looks_non_academic(ref) is False

    def test_no_url_no_venue_is_academic(self):
        ref = make_ref(url=None, venue=None)
        assert _looks_non_academic(ref) is False


# ---------------------------------------------------------------------------
# _determine_verdict
# ---------------------------------------------------------------------------

class TestDetermineVerdict:

    # RETRACTED

    def test_retracted_overrides_everything(self):
        ref = make_ref(doi="10.1/x")
        result = make_verification_result(ref, retracted=True)
        assert _determine_verdict(result) == RETRACTED

    # VERIFIED via identifier

    def test_crossref_high_similarity_verified(self):
        ref = make_ref(doi="10.1038/nature14539")
        result = make_verification_result(
            ref,
            source=VerificationSource.CROSSREF,
            found=True,
            title_similarity=0.95,
            author_match=True,
        )
        assert _determine_verdict(result) == VERIFIED

    def test_arxiv_high_similarity_verified(self):
        ref = make_ref(arxiv_id="1706.03762")
        result = make_verification_result(
            ref,
            source=VerificationSource.ARXIV,
            found=True,
            title_similarity=1.0,
            author_match=None,  # et al. — doesn't matter for identifier path
        )
        assert _determine_verdict(result) == VERIFIED

    def test_identifier_low_similarity_needs_review(self):
        ref = make_ref(doi="10.1/x")
        result = make_verification_result(
            ref,
            source=VerificationSource.CROSSREF,
            found=True,
            title_similarity=0.50,
        )
        assert _determine_verdict(result) == NEEDS_REVIEW

    # VERIFIED via search

    def test_openalex_high_sim_author_match_verified(self):
        ref = make_ref()
        result = make_verification_result(
            ref,
            source=VerificationSource.OPENALEX,
            found=True,
            title_similarity=0.95,
            author_match=True,
        )
        assert _determine_verdict(result) == VERIFIED

    # LIKELY_REAL

    def test_openalex_high_sim_et_al_likely_real(self):
        """title matches but author_match is None (et al.) → LIKELY_REAL"""
        ref = make_ref(authors=["Zirui Guo et al."])
        result = make_verification_result(
            ref,
            source=VerificationSource.OPENALEX,
            found=True,
            title_similarity=1.0,
            author_match=None,
        )
        assert _determine_verdict(result) == LIKELY_REAL

    # NEEDS_REVIEW

    def test_search_high_sim_author_mismatch_needs_review(self):
        ref = make_ref()
        result = make_verification_result(
            ref,
            source=VerificationSource.OPENALEX,
            found=True,
            title_similarity=0.95,
            author_match=False,
        )
        assert _determine_verdict(result) == NEEDS_REVIEW

    def test_search_low_sim_needs_review(self):
        ref = make_ref()
        result = make_verification_result(
            ref,
            source=VerificationSource.OPENALEX,
            found=True,
            title_similarity=0.60,
        )
        assert _determine_verdict(result) == NEEDS_REVIEW

    # UNVERIFIABLE

    def test_not_found_with_url_unverifiable(self):
        """Has a URL — blog or PDF link — not a failure to find."""
        ref = make_ref(url="https://trec-car.cs.unh.edu/paper.pdf")
        result = make_verification_result(ref, found=False)
        result.source_results[0].title_similarity = None
        assert _determine_verdict(result) == UNVERIFIABLE

    def test_github_url_unverifiable(self):
        ref = make_ref(url="https://github.com/prometheus-eval/prometheus-eval")
        result = make_verification_result(ref, found=False)
        result.source_results[0].title_similarity = None
        assert _determine_verdict(result) == UNVERIFIABLE

    def test_book_venue_unverifiable(self):
        ref = make_ref(venue="MIT Press", url=None)
        result = make_verification_result(ref, found=False)
        result.source_results[0].title_similarity = None
        assert _determine_verdict(result) == UNVERIFIABLE

    # LIKELY_HALLUCINATED

    def test_nothing_found_no_url_academic_venue_hallucinated(self):
        ref = make_ref(venue="Nature Machine Intelligence", url=None)
        result = make_verification_result(ref, found=False)
        result.source_results[0].title_similarity = None
        assert _determine_verdict(result) == LIKELY_HALLUCINATED

    def test_citeguard_unpublished_hallucinated(self):
        ref = make_ref(
            title="CiteGuard: Self-supervised agents crosschecking hallucinated references",
            authors=["Danilo Toapanta"],
            year=None,
            url=None,
            venue=None,
        )
        result = make_verification_result(ref, found=False)
        result.source_results[0].title_similarity = None
        assert _determine_verdict(result) == LIKELY_HALLUCINATED

    # WEB_SEARCH rules

    def test_web_search_high_sim_likely_real(self):
        ref = make_ref()
        result = make_verification_result(
            ref,
            source=VerificationSource.WEB_SEARCH,
            found=True,
            title_similarity=0.92,
            author_match=None,
        )
        assert _determine_verdict(result) == LIKELY_REAL

    def test_web_search_high_sim_never_verified(self):
        ref = make_ref()
        result = make_verification_result(
            ref,
            source=VerificationSource.WEB_SEARCH,
            found=True,
            title_similarity=0.99,
            author_match=None,
        )
        assert _determine_verdict(result) != VERIFIED

    def test_web_search_low_sim_needs_review(self):
        ref = make_ref()
        result = make_verification_result(
            ref,
            source=VerificationSource.WEB_SEARCH,
            found=True,
            title_similarity=0.60,
            author_match=None,
        )
        assert _determine_verdict(result) == NEEDS_REVIEW

    def test_web_search_not_found_hallucinated(self):
        ref = make_ref()
        result = make_verification_result(
            ref,
            source=VerificationSource.WEB_SEARCH,
            found=False,
            title_similarity=None,
        )
        assert _determine_verdict(result) == LIKELY_HALLUCINATED


# ---------------------------------------------------------------------------
# merge_results_node
# ---------------------------------------------------------------------------

class TestMergeResultsNode:

    @pytest.mark.asyncio
    async def test_combines_all_three_lists(self):
        ref_doi = make_ref("DOI Paper", doi="10.1/x")
        ref_arxiv = make_ref("arXiv Paper", arxiv_id="1706.03762")
        ref_search = make_ref("Search Paper")

        state = {
            "doi_results": [make_verification_result(ref_doi, VerificationSource.CROSSREF)],
            "arxiv_results": [make_verification_result(ref_arxiv, VerificationSource.ARXIV)],
            "search_results": [make_verification_result(ref_search, VerificationSource.OPENALEX)],
        }
        result = await merge_results_node(state)
        assert len(result["merged_results"]) == 3

    @pytest.mark.asyncio
    async def test_deduplicates_by_raw_reference(self):
        """
        Holoeval scenario: arXiv check failed → queued for search.
        Same paper appears in both arxiv_results and search_results.
        Should produce one merged result with both source signals.
        """
        ref = make_ref("Holoeval", arxiv_id="2310.14746")
        arxiv_result = make_verification_result(
            ref, VerificationSource.ARXIV, found=True, title_similarity=0.45
        )
        search_result = make_verification_result(
            ref, VerificationSource.OPENALEX, found=False
        )

        state = {
            "doi_results": [],
            "arxiv_results": [arxiv_result],
            "search_results": [search_result],
        }
        result = await merge_results_node(state)

        assert len(result["merged_results"]) == 1
        merged = result["merged_results"][0]
        assert len(merged.source_results) == 2
        sources = {sr.source for sr in merged.source_results}
        assert VerificationSource.ARXIV in sources
        assert VerificationSource.OPENALEX in sources

    @pytest.mark.asyncio
    async def test_empty_state_returns_empty(self):
        state = {"doi_results": [], "arxiv_results": [], "search_results": []}
        result = await merge_results_node(state)
        assert result["merged_results"] == []

    @pytest.mark.asyncio
    async def test_missing_keys_handled_gracefully(self):
        """State missing some result keys — should not crash."""
        state = {"doi_results": []}
        result = await merge_results_node(state)
        assert result["merged_results"] == []


# ---------------------------------------------------------------------------
# score_node
# ---------------------------------------------------------------------------

class TestScoreNode:

    @pytest.mark.asyncio
    async def test_assigns_verdict_to_each_result(self):
        refs = [make_ref(f"Paper {i}") for i in range(3)]
        merged = [
            make_verification_result(refs[0], VerificationSource.CROSSREF, True, 0.95),
            make_verification_result(refs[1], VerificationSource.OPENALEX, False),
            make_verification_result(refs[2], VerificationSource.ARXIV, True, 1.0),
        ]
        state = {"merged_results": merged}
        result = await score_node(state)

        scored = result["scored_references"]
        assert len(scored) == 3
        assert all(r.verdict is not None for r in scored)

    @pytest.mark.asyncio
    async def test_verified_crossref_paper(self):
        ref = make_ref(doi="10.1038/nature14539")
        merged = [make_verification_result(ref, VerificationSource.CROSSREF, True, 0.95)]
        result = await score_node({"merged_results": merged})
        assert result["scored_references"][0].verdict == VERIFIED

    @pytest.mark.asyncio
    async def test_likely_hallucinated_not_found(self):
        ref = make_ref(
            title="Quantum Pasta Optimization",
            venue="Nature Machine Intelligence",
            url=None,
        )
        vr = make_verification_result(ref, found=False)
        vr.source_results[0].title_similarity = None
        result = await score_node({"merged_results": [vr]})
        assert result["scored_references"][0].verdict == LIKELY_HALLUCINATED

    @pytest.mark.asyncio
    async def test_retracted_paper_flagged(self):
        ref = make_ref(doi="10.1/retracted")
        merged = [make_verification_result(ref, retracted=True)]
        result = await score_node({"merged_results": merged})
        assert result["scored_references"][0].verdict == RETRACTED

    @pytest.mark.asyncio
    async def test_empty_merged_results(self):
        result = await score_node({"merged_results": []})
        assert result["scored_references"] == []

    @pytest.mark.asyncio
    async def test_verdicts_for_your_16_references(self):
        """
        Smoke test based on the actual pipeline results we've seen.
        Uses realistic signal combinations from our sample_references_2.md run.
        """
        # Kartawijaya [87] — DOI, Crossref confirmed
        kartawijaya = make_verification_result(
            make_ref("Improving Students Writing", doi="10.22216/jcc.2018.v3i3.3429"),
            VerificationSource.CROSSREF, True, 0.90, True
        )

        # LightRAG [80] — arXiv confirmed, et al.
        lightrag = make_verification_result(
            make_ref("LightRAG", arxiv_id="2410.05779"),
            VerificationSource.ARXIV, True, 1.0, None
        )

        # Holoeval [95] — arXiv wrong paper, OpenAlex not found
        holoeval_ref = make_ref("Holoeval", arxiv_id="2310.14746", venue="arXiv preprint")
        holoeval = VerificationResult(
            reference=holoeval_ref,
            sources_checked=[VerificationSource.ARXIV, VerificationSource.OPENALEX],
            source_results=[
                make_source_result(VerificationSource.ARXIV, True, 0.45, None),
                make_source_result(VerificationSource.OPENALEX, False, None, None),
            ],
        )

        # CiteGuard [94] — nothing found, no url
        citeguard = make_verification_result(
            make_ref("CiteGuard", url=None, venue=None),
            VerificationSource.OPENALEX, False
        )
        citeguard.source_results[0].title_similarity = None

        # GPT-4o blog [90] — has URL, not found
        gpt4o = make_verification_result(
            make_ref("GPT-4o mini", url="https://openai.com/blog/gpt4o"),
            VerificationSource.OPENALEX, False
        )
        gpt4o.source_results[0].title_similarity = None

        # Latour book [92] — MIT Press venue
        latour = make_verification_result(
            make_ref("Drawing things together", venue="MIT Press", url=None),
            VerificationSource.OPENALEX, False
        )
        latour.source_results[0].title_similarity = None

        state = {"merged_results": [kartawijaya, lightrag, holoeval, citeguard, gpt4o, latour]}
        result = await score_node(state)
        scored = result["scored_references"]

        verdicts = {r.reference.title: r.verdict for r in scored}

        assert verdicts["Improving Students Writing"] == VERIFIED
        assert verdicts["LightRAG"] == VERIFIED      # identifier path, title matches
        assert verdicts["Holoeval"] == NEEDS_REVIEW  # arXiv found wrong paper (0.45)
        assert verdicts["CiteGuard"] == LIKELY_HALLUCINATED
        assert verdicts["GPT-4o mini"] == UNVERIFIABLE
        assert verdicts["Drawing things together"] == UNVERIFIABLE