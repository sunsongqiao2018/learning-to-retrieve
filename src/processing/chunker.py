from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.datasets.schema import DocumentChunk, NornalizedContextSentence


def chunk_contexts(
    contexts: list[NornalizedContextSentence], chunk_size: int = 300, chunk_overlap: int = 80
) -> list[DocumentChunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks: list[DocumentChunk] = []
    for context in contexts:
        parts = splitter.split_text(context.text)
        for idx, text in enumerate(parts):
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{context.example_id}_title_{context.title}_sent_{context.sent_id}_chunk_{idx}",
                    text=text,
                    metadata={
                        "dataset": context.dataset,
                        "split": context.split,
                        "example_id": context.example_id,
                    },
                )
            )
    return chunks
