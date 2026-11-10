"""Recorded authored smoke responses for schema illustration; timings are incidental."""

SEARCH_EXAMPLE = {
    "method": "bm25",
    "selected_strategy": "bm25",
    "top_k": 1,
    "documents": [
        {
            "document_id": "d2",
            "rank": 1,
            "score": 1.3346109202546488,
            "title": "Database error",
            "text": "ERR_42 indicates a database connection timeout.",
            "metadata": {"dataset": "synthetic-smoke"},
        }
    ],
    "score_type": "bm25_okapi",
    "ranking_explanation": "Lexical BM25 Okapi; higher lexical score ranks first. "
    "Ties break by document ID.",
    "routing": None,
    "candidate_count": 0,
    "initialized_resources": ["corpus", "bm25"],
    "latency": {
        "initialization_ms": 0.6162999197840691,
        "retrieval_ms": 0.11399993672966957,
        "rerank_ms": 0.0,
        "serialization_ms": 0.00380026176571846,
        "queue_ms": 0.005200039595365524,
        "total_service_ms": 0.7552001625299454,
    },
    "request_id": "de159ff67764412fb72f2b87de702cfc",
}

COMPARE_EXAMPLE = {
    "request_id": "2a3addb623f34f97af98f2c185d2dcc0",
    "results": [
        {
            "method": "bm25",
            "selected_strategy": "bm25",
            "top_k": 1,
            "documents": [
                {
                    "document_id": "d2",
                    "rank": 1,
                    "score": 1.3346109202546488,
                    "title": "Database error",
                    "text": "ERR_42 indicates a database connection timeout.",
                    "metadata": {"dataset": "synthetic-smoke"},
                }
            ],
            "score_type": "bm25_okapi",
            "ranking_explanation": "Lexical BM25 Okapi; higher lexical score "
            "ranks first. Ties break by document ID.",
            "routing": None,
            "candidate_count": 0,
            "initialized_resources": [],
            "latency": {
                "initialization_ms": 0.0,
                "retrieval_ms": 0.06309989839792252,
                "rerank_ms": 0.0,
                "serialization_ms": 0.0029997900128364563,
                "queue_ms": 0.0,
                "total_service_ms": 0.06669992581009865,
            },
        }
    ],
    "agreement": [],
    "queue_ms": 0.00380026176571846,
    "total_service_ms": 0.08299993351101875,
    "interpretation": "Rank overlap/order only. Scores from different methods are "
    "not directly comparable; no relevance metrics are "
    "computed.",
}

ERROR_EXAMPLE = {
    "request_id": "332bad96eb214a73b3ab2a0ca68ed594",
    "error": {
        "code": "invalid_request",
        "message": "Request validation failed.",
        "details": [
            {
                "field": "body.query",
                "code": "string_too_short",
                "message": "String should have at least 1 character",
            }
        ],
    },
}
