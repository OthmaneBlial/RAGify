from backend.modules.rag.retrieval import RetrievalResult


def test_citation_highlight_focus(rag_pipeline):
    top_result = RetrievalResult(
        content="Domestic - La Liga: 36 titles - Copa del Rey: 20 titles",
        score=0.92,
        metadata={
            "document_id": "doc-1",
            "document_title": "file.txt",
            "paragraph_id": "para-1",
            "knowledge_base_id": "kb-1",
            "paragraph_excerpt": "Domestic - La Liga: 36 titles - Copa del Rey: 20 titles",
        },
        source="file.txt",
    )

    lower_result = RetrievalResult(
        content="Santiago Bernabéu presidency from 1943 transformed the club",
        score=0.55,
        metadata={
            "document_id": "doc-2",
            "document_title": "file-2.txt",
            "paragraph_id": "para-2",
            "knowledge_base_id": "kb-1",
            "paragraph_excerpt": "Santiago Bernabéu presidency from 1943 transformed the club",
        },
        source="file-2.txt",
    )

    citations = rag_pipeline._build_citations(
        [top_result, lower_result],
        "Real Madrid have won 36 La Liga titles",
        "How many La Liga won by Real Madrid?",
    )

    assert len(citations) == 1
    best_citation = citations[0]
    assert best_citation["document_title"] == "file.txt"
    assert "<mark>36</mark>" in best_citation["snippet"]
    assert "<mark>Liga</mark>" in best_citation["snippet"]


def test_query_terms_prioritized(rag_pipeline):
    stars_section = RetrievalResult(
        content="Current stars: Kylian Mbappé, Jude Bellingham, Vinícius Júnior, Rodrygo",
        score=0.55,
        metadata={
            "document_id": "doc-stars",
            "document_title": "file.txt",
            "paragraph_id": "para-stars",
            "knowledge_base_id": "kb-1",
            "paragraph_excerpt": "Current stars: Kylian Mbappé, Jude Bellingham, Vinícius Júnior, Rodrygo",
        },
        source="file.txt",
    )

    highlights_section = RetrievalResult(
        content="Recent Highlights - Strong Champions League campaigns with Mbappé and Bellingham",
        score=0.7,
        metadata={
            "document_id": "doc-highlights",
            "document_title": "file.txt",
            "paragraph_id": "para-highlights",
            "knowledge_base_id": "kb-1",
            "paragraph_excerpt": "Recent Highlights - Strong Champions League campaigns with Mbappé and Bellingham",
        },
        source="file.txt",
    )

    citations = rag_pipeline._build_citations(
        [highlights_section, stars_section],
        "Current stars at Real Madrid include Kylian Mbappé and Jude Bellingham.",
        "Who are the current stars in Real Madrid?",
    )

    assert len(citations) == 1
    selected = citations[0]
    assert selected["paragraph_id"] == "para-stars"
    assert "<mark>Current stars</mark>" in selected["snippet"] or "<mark>Current</mark>" in selected["snippet"]


def test_no_citation_when_no_match(rag_pipeline):
    unrelated_context = RetrievalResult(
        content="Stadium information and ticket office hours",
        score=0.35,
        metadata={
            "document_id": "doc-1",
            "document_title": "file.txt",
            "paragraph_id": "para-1",
            "knowledge_base_id": "kb-1",
            "paragraph_excerpt": "Stadium information and ticket office hours",
        },
        source="file.txt",
    )

    citations = rag_pipeline._build_citations([
        unrelated_context
    ], None, "Who won world cup 2024")

    assert citations == []


def test_no_citation_when_disclaimer_answer(rag_pipeline):
    unrelated_context = RetrievalResult(
        content="Club honors and domestic trophies",
        score=0.5,
        metadata={
            "document_id": "doc-2",
            "document_title": "file.txt",
            "paragraph_id": "para-honors",
            "knowledge_base_id": "kb-1",
            "paragraph_excerpt": "Club honors and domestic trophies",
        },
        source="file.txt",
    )

    citations = rag_pipeline._build_citations(
        [unrelated_context],
        "The provided context does not contain information about the 2022 FIFA World Cup winner.",
        "Who won world cup 2022?",
    )

    assert citations == []
