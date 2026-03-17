"""
OpenAlex Diagnostic Tests

Comprehensive investigation of OpenAlex coverage and reliability.
Shows raw API responses, what _search returns, what _best_match picks,
and the final verdict for every reference type we care about.

Run with:
    pytest tests/agents/tools/test_openalex_diagnostic.py -v -m slow -s
"""

import asyncio
import pytest
import httpx
from rapidfuzz import fuzz

from app.models.schemas import ReferenceResult
from app.agents.tools.verifiers.openalex import (
    OpenAlexVerifier,
    _normalize_title,
    _best_match,
    TITLE_MATCH_THRESHOLD,
)

SEPARATOR = "=" * 70


def print_section(title: str):
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)


def print_candidates(candidates: list[dict], norm_cited: str):
    if not candidates:
        print("  → NO RESULTS RETURNED")
        return
    print(f"  → {len(candidates)} candidates:")
    for i, c in enumerate(candidates):
        title = c.get("title", "") or ""
        year = c.get("publication_year", "?")
        doi = c.get("doi", "")
        sim = fuzz.ratio(_normalize_title(title), norm_cited) / 100.0
        passes = "✅" if sim >= TITLE_MATCH_THRESHOLD else "❌"
        print(f"  [{i}] {passes} sim={sim:.3f} | {title[:60]}")
        print(f"       year={year} | doi={doi}")


@pytest.mark.slow
class TestOpenAlexDiagnostic:

    @pytest.fixture(autouse=True)
    def setup(self):
        self.verifier = OpenAlexVerifier()

    # -----------------------------------------------------------------------
    # Category 1: Papers that SHOULD be found (verified in previous runs)
    # -----------------------------------------------------------------------

    async def test_cat1_verified_papers(self):
        """Papers we know OpenAlex finds — baseline sanity check."""
        print_section("CATEGORY 1: Papers that should be VERIFIED")

        cases = [
            {
                "title": "Generating biographies on Wikipedia: The impact of gender bias on the retrieval-based generation of women biographies",
                "authors": ["Angela Fan", "Claire Gardent"],
                "year": 2022,
                "note": "ACL 2022 — previously got title_sim=1.0",
            },
            {
                "title": "The concept of information overload: A review of literature from organization science, accounting, marketing, MIS, and related disciplines",
                "authors": ["Martin J Eppler", "Jeanne Mengis"],
                "year": 2004,
                "note": "TACL 2004 — previously got title_sim=1.0",
            },
            {
                "title": "A structured review of the validity of BLEU",
                "authors": ["Ehud Reiter"],
                "year": 2018,
                "note": "CL 2018 — previously got title_sim=1.0",
            },
            {
                "title": "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
                "authors": ["Nils Reimers", "Iryna Gurevych"],
                "year": 2019,
                "note": "EMNLP 2019 — has DOI, used as search test",
            },
        ]

        for case in cases:
            ref = ReferenceResult(
                title=case["title"],
                authors=case["authors"],
                year=case["year"],
                raw_reference=case["title"],
            )
            candidates = await self.verifier._search(ref.title, ref.authors)
            best = _best_match(ref, candidates)
            result = await self.verifier.verify(ref)
            sr = result.source_results[0]

            print(f"\n  [{case['note']}]")
            print(f"  title:   {case['title'][:60]}")
            print(
                f"  found:   {sr.found} | sim={sr.title_similarity} | author={sr.author_match}"
            )
            if best:
                print(f"  matched: {best.get('title', '')[:60]}")
            else:
                print(f"  matched: NONE — {len(candidates)} candidates below threshold")

    # -----------------------------------------------------------------------
    # Category 2: Papers flagged LIKELY_HALLUCINATED (need investigation)
    # -----------------------------------------------------------------------

    async def test_cat2_false_negatives(self):
        """Papers we suspect are real but OpenAlex didn't find."""
        print_section("CATEGORY 2: Suspected FALSE NEGATIVES")

        cases = [
            {
                "title": "InPars: Unsupervised dataset generation for information retrieval",
                "authors": ["Luiz Bonifacio", "Hugo Abonizio", "Rodrigo Nogueira"],
                "year": 2022,
                "note": "ACM SIGIR 2022 — real paper, OpenAlex inconsistent",
            },
            {
                "title": "Efficient Memory Management for Large Language Model Serving with PagedAttention",
                "authors": ["Woosuk Kwon"],
                "year": 2023,
                "note": "SOSP 2023 — title was E#cient in PDF",
            },
            {
                "title": "PAQ: 65 million probably-asked questions and what you can do with them",
                "authors": ["Patrick Lewis"],
                "year": 2021,
                "note": "TACL 2021 — no arXiv ID",
            },
            {
                "title": "Finetuned language models are zero-shot learners",
                "authors": ["Jason Wei"],
                "year": 2021,
                "note": "FLAN paper — arXiv ID missed by LLM",
            },
            {
                "title": "Fact or Fiction: Verifying Scientific Claims",
                "authors": ["David Wadden"],
                "year": 2020,
                "note": "EMNLP 2020 — DOI corrupted in PDF",
            },
            {
                "title": "GPT-J-6B: A 6 billion parameter autoregressive language model",
                "authors": ["Ben Wang", "Aran Komatsuzaki"],
                "year": 2021,
                "note": "EleutherAI blog/report — grey literature",
            },
        ]

        for case in cases:
            ref = ReferenceResult(
                title=case["title"],
                authors=case["authors"],
                year=case["year"],
                raw_reference=case["title"],
            )
            norm_cited = _normalize_title(case["title"])

            # Get raw candidates
            candidates = await self.verifier._search(ref.title, ref.authors)
            best = _best_match(ref, candidates)
            result = await self.verifier.verify(ref)
            sr = result.source_results[0]

            print(f"\n  [{case['note']}]")
            print(f"  title:  {case['title'][:60]}")
            print(
                f"  found:  {sr.found} | sim={sr.title_similarity} | author={sr.author_match}"
            )
            print_candidates(candidates, norm_cited)
            if best:
                print(f"  → best_match picked: {best.get('title', '')[:60]}")
            else:
                print(f"  → best_match returned NONE")

    # -----------------------------------------------------------------------
    # Category 3: Test the subtitle-colon search strategy
    # -----------------------------------------------------------------------

    async def test_cat3_colon_title_strategy(self):
        """
        Does stripping the subtitle (before colon) help for colon titles?
        Tests our current _search implementation.
        """
        print_section("CATEGORY 3: Colon title search strategy")

        colon_titles = [
            "FLAIR: An Easy-to-Use Framework for State-of-the-Art NLP",
            "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
            "InPars: Unsupervised dataset generation for information retrieval",
            "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
            "GPT-J-6B: A 6 billion parameter autoregressive language model",
        ]

        async with httpx.AsyncClient(follow_redirects=True) as client:
            for title in colon_titles:
                short_title = title.split(":")[0].strip()
                full_title = title

                # Test 1: filter with short title
                r1 = await client.get(
                    "https://api.openalex.org/works",
                    params={
                        "filter": f"title.search:{short_title}",
                        "per_page": 3,
                        "mailto": "citeguard@test.com",
                        "select": "title,publication_year",
                    },
                )
                results1 = r1.json().get("results", []) if r1.status_code == 200 else []

                # Test 2: search with full title + first author
                r2 = await client.get(
                    "https://api.openalex.org/works",
                    params={
                        "search": full_title,
                        "per_page": 3,
                        "mailto": "citeguard@test.com",
                        "select": "title,publication_year",
                    },
                )
                results2 = r2.json().get("results", []) if r2.status_code == 200 else []

                norm = _normalize_title(title)
                print(f"\n  Title: {title[:60]}")
                print(f"  Short: {short_title}")

                best1_title = results1[0].get("title", "NONE") if results1 else "NONE"
                sim1 = (
                    fuzz.ratio(_normalize_title(best1_title), norm) / 100.0
                    if results1
                    else 0
                )
                print(
                    f"  filter short → top: {best1_title[:50]} | sim={sim1:.3f} {'✅' if sim1 >= TITLE_MATCH_THRESHOLD else '❌'}"
                )

                best2_title = results2[0].get("title", "NONE") if results2 else "NONE"
                sim2 = (
                    fuzz.ratio(_normalize_title(best2_title), norm) / 100.0
                    if results2
                    else 0
                )
                print(
                    f"  search full  → top: {best2_title[:50]} | sim={sim2:.3f} {'✅' if sim2 >= TITLE_MATCH_THRESHOLD else '❌'}"
                )

    # -----------------------------------------------------------------------
    # Category 4: OpenAlex search parameter comparison
    # -----------------------------------------------------------------------

    async def test_cat4_search_params_comparison(self):
        """
        Compare different OpenAlex search strategies for the same paper.
        Which approach gives the best results?
        """
        print_section("CATEGORY 4: Search strategy comparison")

        test_paper = "Efficient Memory Management for Large Language Model Serving with PagedAttention"
        test_paper_clean = "Efficient Memory Management for Large Language Model Serving with PagedAttention"
        author = "Kwon"

        strategies = {
            "search (original title)": {"search": test_paper},
            "search (clean title)": {"search": test_paper_clean},
            "search (title + author)": {"search": f"{test_paper_clean} {author}"},
            "filter title.search": {
                "filter": f"title.search:{test_paper_clean.split(':')[0]}"
            },
        }

        async with httpx.AsyncClient(follow_redirects=True) as client:
            norm = _normalize_title(test_paper_clean)
            print(f"\n  Paper: {test_paper_clean[:60]}")

            for strategy_name, params in strategies.items():
                params["per_page"] = 3
                params["mailto"] = "citeguard@test.com"
                params["select"] = "title,publication_year,doi"

                r = await client.get("https://api.openalex.org/works", params=params)
                results = r.json().get("results", []) if r.status_code == 200 else []
                total = (
                    r.json().get("meta", {}).get("count", 0)
                    if r.status_code == 200
                    else 0
                )

                if results:
                    top = results[0].get("title", "NONE")
                    sim = fuzz.ratio(_normalize_title(top), norm) / 100.0
                    status = "✅" if sim >= TITLE_MATCH_THRESHOLD else "❌"
                    print(f"\n  [{strategy_name}] total={total}")
                    print(f"    top: {top[:60]} | sim={sim:.3f} {status}")
                else:
                    print(
                        f"\n  [{strategy_name}] → NO RESULTS (status={r.status_code})"
                    )

    # -----------------------------------------------------------------------
    # Category 5: Consistency test — run same queries multiple times
    # -----------------------------------------------------------------------

    async def test_cat5_consistency(self):
        """
        Run the same query 3 times to check if OpenAlex is consistent.
        Non-deterministic results indicate a reliability problem.
        """
        print_section("CATEGORY 5: Consistency check (3 runs each)")

        papers = [
            "InPars: Unsupervised dataset generation for information retrieval",
            "FLAIR: An Easy-to-Use Framework for State-of-the-Art NLP",
        ]

        for title in papers:
            ref = ReferenceResult(
                title=title,
                authors=["Test Author"],
                year=2022,
                raw_reference=title,
            )
            found_results = []
            norm = _normalize_title(title)

            print(f"\n  Paper: {title[:60]}")

            for run in range(3):
                candidates = await self.verifier._search(ref.title, ref.authors)
                best = _best_match(ref, candidates)
                found = best is not None
                sim = best.get("_title_similarity", 0) if best else 0
                found_results.append(found)
                print(
                    f"    Run {run+1}: found={found} | sim={sim:.3f} | candidates={len(candidates)}"
                )

            consistent = len(set(found_results)) == 1
            print(
                f"    Consistent: {'✅ YES' if consistent else '❌ NO — non-deterministic!'}"
            )

    # -----------------------------------------------------------------------
    # Category 6: Threshold sensitivity
    # -----------------------------------------------------------------------

    async def test_cat6_threshold_sensitivity(self):
        """
        What happens if we lower the threshold from 0.85 to 0.80?
        Does it help catch more real papers without too many false positives?
        """
        print_section("CATEGORY 6: Threshold sensitivity analysis")

        cases_should_find = [
            (
                "Prometheus 2: An Open-Source Language Model Specialised in Evaluating Other LLMs",
                "Seungone Kim",
                2024,
                "Prometheus 2 (title slightly different)",
            ),
        ]
        cases_should_not_find = [
            (
                "Quantum Neural Transformers for Distributed Hyperparameter Optimization",
                "A. Fake",
                2023,
                "Fake paper — should NOT be found at any threshold",
            ),
        ]

        thresholds = [0.75, 0.80, 0.85, 0.90]

        for title, author, year, note in cases_should_find + cases_should_not_find:
            ref = ReferenceResult(
                title=title, authors=[author], year=year, raw_reference=title
            )
            candidates = await self.verifier._search(ref.title, ref.authors)
            norm = _normalize_title(title)

            print(f"\n  [{note}]")
            print(f"  title: {title[:60]}")

            if candidates:
                best_title = candidates[0].get("title", "")
                sim = fuzz.ratio(_normalize_title(best_title), norm) / 100.0
                print(f"  top candidate: {best_title[:60]} | sim={sim:.3f}")
                for t in thresholds:
                    found = sim >= t
                    print(
                        f"    threshold={t}: {'✅ FOUND' if found else '❌ NOT FOUND'}"
                    )
            else:
                print("  → no candidates returned")
