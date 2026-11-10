from dataclasses import dataclass


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    text: str

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("Document ID must be a nonempty string")
        if not isinstance(self.title, str) or not isinstance(self.text, str):
            raise ValueError("Document title and text must be strings")


@dataclass(frozen=True)
class Query:
    id: str
    text: str

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("Query ID must be a nonempty string")
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("Query text must be a nonempty string")


@dataclass
class Dataset:
    documents: dict[str, Document]
    queries: dict[str, Query]
    qrels: dict[str, dict[str, int]]
    metadata: dict
