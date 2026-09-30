from pathlib import Path
import hashlib
import re
import os
from dotenv import load_dotenv

from langchain_core.documents import Document
from embedder import SentenceTransformerEmbeddings
from langchain_astradb import AstraDBVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ============================================================
# CONFIGURATION
# ============================================================
load_dotenv(override=True)

BASE_DIR = Path(__file__).resolve().parent.parent

DOCS_DIR = BASE_DIR / "docs" / "vinayak_rahasya"

COLLECTION_NAME = "rahasya"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

ASTRA_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")

if not ASTRA_TOKEN:
    raise EnvironmentError(
        "ASTRA_DB_APPLICATION_TOKEN environment variable is not set."
    )

if not ASTRA_ENDPOINT:
    raise EnvironmentError(
        "ASTRA_DB_API_ENDPOINT environment variable is not set."
    )


# ============================================================
# SOURCE INFORMATION
# ============================================================

SOURCE_NAME = "Vinayak Rahasya"

SOURCE_TYPE = "traditional_text"

DOCUMENT_TYPE = "rahasya"

EDITOR = "Kunnam Brahmashri V. Vishwanatha Sarma"

PUBLISHED_BY = "Sri G. Ramaswamy"

AUTHORITY = "traditional_text"


# ============================================================
# TEXT SPLITTER
# ============================================================

# Vinayak Rahasya files may contain relatively long sections.
#
# Chunk size is deliberately moderate so that:
# - individual philosophical discussions remain together
# - retrieval does not return excessively large passages
# - citations remain reasonably precise
#
# overlap helps preserve context between chunks.

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=200,
    separators=[
        "\n\n",
        "\n",
        ". ",
        "? ",
        "! ",
        "; ",
        ", ",
        " ",
    ],
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_text(text: str) -> str:
    """
    Basic text normalization.

    This does NOT alter Sanskrit terminology or rewrite the source.
    It only normalizes whitespace introduced by the text files.
    """

    # Normalize Windows / Mac line endings
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Remove trailing spaces
    text = "\n".join(line.rstrip() for line in text.splitlines())

    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def get_document_title(file_path: Path) -> str:
    """
    Convert filename into a readable document title.

    Example:
        Definition_and_Examination_of_Philosophical_Doctrines.txt

    becomes:
        Definition and Examination of Philosophical Doctrines
    """

    title = file_path.stem

    # Replace underscores with spaces
    title = title.replace("_", " ")

    # Remove excessive whitespace
    title = re.sub(r"\s+", " ", title)

    return title.strip()


def create_chunk_id(
    source: str,
    document_title: str,
    chunk_index: int,
    content: str,
) -> str:
    """
    Create a deterministic ID for every chunk.
    """

    raw = (
        f"{source}|"
        f"{document_title}|"
        f"{chunk_index}|"
        f"{content}"
    )

    return hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()


def create_citation(
    document_title: str,
    chunk_index: int,
) -> str:
    """
    Create a human-readable citation.

    Example:
        (Vinayak Rahasya, Description of Ganapati's Form, Section 2)
    """

    return (
        f"({SOURCE_NAME}, "
        f"{document_title}, "
        f"Section {chunk_index + 1})"
    )


# ============================================================
# LOAD DOCUMENTS
# ============================================================

def load_documents():
    """
    Read all .txt files from docs/vinayak_rahasya and
    convert them into LangChain Documents.
    """

    if not DOCS_DIR.exists():
        raise FileNotFoundError(
            f"Directory not found: {DOCS_DIR}"
        )

    files = sorted(DOCS_DIR.glob("*.txt"))

    if not files:
        raise FileNotFoundError(
            f"No .txt files found in: {DOCS_DIR}"
        )

    print("=" * 70)
    print("VINAYAK RAHASYA INGESTION")
    print("=" * 70)

    print(f"Source directory : {DOCS_DIR}")
    print(f"Collection       : {COLLECTION_NAME}")
    print(f"Text files found : {len(files)}")
    print()

    documents = []

    for file_path in files:

        print(f"Reading: {file_path.name}")

        try:
            text = file_path.read_text(
                encoding="utf-8"
            )
        except UnicodeDecodeError:
            # Fallback for files that may have been saved
            # with a different encoding.
            text = file_path.read_text(
                encoding="utf-8-sig"
            )

        text = clean_text(text)

        if not text:
            print("  WARNING: Empty file — skipped.")
            continue

        document_title = get_document_title(file_path)

        # ----------------------------------------------------
        # Split the source text into retrieval chunks
        # ----------------------------------------------------

        chunks = text_splitter.split_text(text)

        total_chunks = len(chunks)

        print(
            f"  Title : {document_title}"
        )

        print(
            f"  Chunks: {total_chunks}"
        )

        # ----------------------------------------------------
        # Create LangChain Documents
        # ----------------------------------------------------

        for chunk_index, chunk in enumerate(chunks):

            chunk_id = create_chunk_id(
                source=SOURCE_NAME,
                document_title=document_title,
                chunk_index=chunk_index,
                content=chunk,
            )

            citation = create_citation(
                document_title=document_title,
                chunk_index=chunk_index,
            )

            metadata = {
                # Collection
                "collection": COLLECTION_NAME,

                # Source classification
                "source_type": SOURCE_TYPE,
                "document_type": DOCUMENT_TYPE,

                # Source
                "source": SOURCE_NAME,
                "source_name": SOURCE_NAME,
                "text_source": SOURCE_NAME,

                # Document information
                "document_title": document_title,
                "file_name": file_path.name,
                "file_path": str(
                    file_path.relative_to(BASE_DIR)
                ),

                # Bibliographic information
                "editor": EDITOR,
                "published_by": PUBLISHED_BY,

                # Authority / tradition
                "authority": AUTHORITY,
                "tradition": "Ganapatya",

                # Chunk information
                "chunk_id": chunk_id,
                "chunk_index": chunk_index,
                "total_chunks": total_chunks,

                # Citation
                "citation": citation,

                # Generic structural fields
                "section": document_title,
                "chapter": "",
                "chapter_number": "",
                "chapter_title": "",
                "page_number": "",
            }

            documents.append(
                Document(
                    page_content=chunk,
                    metadata=metadata,
                )
            )

    print()
    print(f"Total chunks prepared: {len(documents)}")

    return documents


# ============================================================
# CREATE EMBEDDING MODEL
# ============================================================

def create_embeddings():

    print()
    print("=" * 70)
    print("LOADING EMBEDDING MODEL")
    print("=" * 70)

    print(
        f"Embedding model: {EMBEDDING_MODEL}"
    )
    
    embeddings = SentenceTransformerEmbeddings(
        model_name=EMBEDDING_MODEL,
        device=None
    )

    print("Embedding model loaded.")

    return embeddings


# ============================================================
# CONNECT TO ASTRA DB
# ============================================================

def create_vector_store(embeddings):

    print()
    print("=" * 70)
    print("CONNECTING TO ASTRA DB")
    print("=" * 70)

    print(
        f"Collection: {COLLECTION_NAME}"
    )

    vector_store = AstraDBVectorStore(
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
        token=ASTRA_TOKEN,
        api_endpoint=ASTRA_ENDPOINT,
    )

    print("Astra DB connection initialized.")

    return vector_store


# ============================================================
# INGEST
# ============================================================

def ingest_documents(
    vector_store,
    documents,
):

    print()
    print("=" * 70)
    print("UPLOADING DOCUMENTS")
    print("=" * 70)

    if not documents:
        print("No documents to upload.")
        return

    total = len(documents)

    print(
        f"Uploading {total} chunks..."
    )

    vector_store.add_documents(
        documents
    )

    print()
    print("Upload completed successfully.")


# ============================================================
# PRINT SUMMARY
# ============================================================

def print_summary(documents):

    source_counts = {}

    for document in documents:

        source = document.metadata.get(
            "source",
            "Unknown"
        )

        source_counts[source] = (
            source_counts.get(source, 0) + 1
        )

    print()
    print("=" * 70)
    print("INGESTION SUMMARY")
    print("=" * 70)

    print(
        f"Collection      : {COLLECTION_NAME}"
    )

    print(
        f"Source          : {SOURCE_NAME}"
    )

    print(
        f"Files processed : "
        f"{len(set(
            d.metadata['file_name']
            for d in documents
        ))}"
    )

    print(
        f"Chunks uploaded : {len(documents)}"
    )

    print(
        f"Editor          : {EDITOR}"
    )

    print(
        f"Published by    : {PUBLISHED_BY}"
    )

    print()

    print("Chunks by source:")

    for source, count in source_counts.items():

        print(
            f"  {source}: {count}"
        )

    print()
    print("Ingestion finished.")
    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main():

    # 1. Read and chunk the text files
    documents = load_documents()

    # 2. Load embedding model
    embeddings = create_embeddings()

    # 3. Connect to Astra DB
    vector_store = create_vector_store(
        embeddings
    )

    # 4. Upload
    ingest_documents(
        vector_store,
        documents,
    )

    # 5. Summary
    print_summary(documents)


if __name__ == "__main__":
    main()