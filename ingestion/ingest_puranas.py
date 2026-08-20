"""
GaneshRAG - Puranas Ingestion Pipeline

Purpose:
    Ingest Ganesh Purana and Mudgal Purana into Astra DB.

Database:
    ganesa_data

Keyspace:
    default_keyspace

Collection:
    puranas

Embedding Model:
    all-MiniLM-L6-v2

Source:
    docs/
        ganesh_puran/
            Ganesh_Puran_Krida_Khand/
            Ganesh_Puran_Upasana_Khand/

        mudgal/
            Mudgal_Puran_Khand_1/
            ...
            Mudgal_Puran_Khand_9/

Pipeline:
    Documents
        ↓
    Chapter Loading
        ↓
    Metadata Enrichment
        ↓
    Chunking
        ↓
    Deterministic IDs
        ↓
    Embeddings
        ↓
    Astra DB
"""


# ============================================================
# 1. IMPORTS
# ============================================================

import os
import re
import hashlib
from pathlib import Path

import dotenv

from loaders.chapter_loader import ChapterLoader
from embedder import SentenceTransformerEmbeddings

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_astradb import AstraDBVectorStore

from astrapy import DataAPIClient


# ============================================================
# 2. CONFIGURATION
# ============================================================

# Load environment variables
dotenv.load_dotenv(override=True)


# ------------------------------------------------------------
# Project paths
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DOCS_DIR = BASE_DIR / "docs"

GANESH_PURAN_DIR = DOCS_DIR / "ganesh_puran"
MUDGAL_PURAN_DIR = DOCS_DIR / "mudgal"


# ------------------------------------------------------------
# Astra DB configuration
# ------------------------------------------------------------

ASTRA_DB_API_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")

# The database should now be your new GaneshRAG database.
#
# IMPORTANT:
# The actual database name is determined by the Astra endpoint.
# This variable is mainly used for validation/documentation.
ASTRA_DB_NAME = os.getenv(
    "ASTRA_DB_NAME",
    "ganesa_data"
)

ASTRA_DB_KEYSPACE = os.getenv(
    "ASTRA_DB_KEYSPACE",
    "default_keyspace"
)

ASTRA_COLLECTION_NAME = "puranas"


# ------------------------------------------------------------
# Embedding configuration
# ------------------------------------------------------------

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

EMBEDDING_VERSION = "v1"


# ------------------------------------------------------------
# Chunking configuration
# ------------------------------------------------------------

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

CHUNKING_VERSION = "v1"


# ------------------------------------------------------------
# Source metadata
# ------------------------------------------------------------

SOURCE_TYPE = "purana"

AUTHORITY = "primary_scripture"

TRADITION = "Ganapatya"

LANGUAGE = "English"


# ============================================================
# 3. VALIDATE ENVIRONMENT
# ============================================================

def validate_environment():
    """
    Validate required environment variables and directories.
    """

    if not ASTRA_DB_API_ENDPOINT:
        raise RuntimeError(
            "ASTRA_DB_API_ENDPOINT is not defined in .env"
        )

    if not ASTRA_DB_APPLICATION_TOKEN:
        raise RuntimeError(
            "ASTRA_DB_APPLICATION_TOKEN is not defined in .env"
        )

    if not DOCS_DIR.exists():
        raise FileNotFoundError(
            f"Documents directory not found: {DOCS_DIR}"
        )

    if not GANESH_PURAN_DIR.exists():
        raise FileNotFoundError(
            f"Ganesh Purana directory not found: {GANESH_PURAN_DIR}"
        )

    if not MUDGAL_PURAN_DIR.exists():
        raise FileNotFoundError(
            f"Mudgal Purana directory not found: {MUDGAL_PURAN_DIR}"
        )

    print("Environment validation successful.")


# ============================================================
# 4. CONNECT TO ASTRA DB
# ============================================================

def connect_to_database():
    """
    Connect to Astra DB using the Data API.

    Returns:
        Database object
    """

    client = DataAPIClient()

    database = client.get_database(
        ASTRA_DB_API_ENDPOINT,
        token=ASTRA_DB_APPLICATION_TOKEN
    )

    try:
        database_name = database.info().name
    except Exception:
        database_name = ASTRA_DB_NAME

    print("\nConnected to Astra DB")
    print(f"Database : {database_name}")
    print(f"Keyspace : {ASTRA_DB_KEYSPACE}")

    return database


# ============================================================
# 5. LOAD CHAPTERS
# ============================================================

def load_puranas():
    """
    Load all chapters using the existing ChapterLoader.

    This preserves the loading approach used in the
    original dataIngestion.ipynb.
    """

    chapter_loader = ChapterLoader()

    paths = {
        str(GANESH_PURAN_DIR): [
            "Ganesh_Puran_Krida_Khand",
            "Ganesh_Puran_Upasana_Khand"
        ],

        str(MUDGAL_PURAN_DIR): [
            f"Mudgal_Puran_Khand_{i}"
            for i in range(1, 10)
        ]
    }

    print("\n" + "=" * 70)
    print("LOADING PURANAS")
    print("=" * 70)

    all_documents = chapter_loader.load_from_paths(paths)

    print(
        f"\nTotal chapter documents loaded: "
        f"{len(all_documents)}"
    )

    if not all_documents:
        raise RuntimeError(
            "No documents were loaded. "
            "Check your docs/ directory structure."
        )

    return all_documents


# ============================================================
# 6. IDENTIFY PURANA
# ============================================================

def identify_purana(book_name: str) -> str:
    """
    Convert the existing 'book' metadata into a cleaner
    high-level source name.

    Examples:
        Ganesh_Puran_Krida_Khand
            -> Ganesh Purana

        Ganesh_Puran_Upasana_Khand
            -> Ganesh Purana

        Mudgal_Puran_Khand_8
            -> Mudgal Purana
    """

    normalized = book_name.lower()

    if "ganesh_puran" in normalized:
        return "Ganesh Purana"

    if "mudgal_puran" in normalized:
        return "Mudgal Purana"

    return book_name


# ============================================================
# 7. EXTRACT KHANDA / SECTION
# ============================================================

def extract_section(book_name: str) -> str:
    """
    Extract Khanda information from the original book name.

    Examples:
        Ganesh_Puran_Krida_Khand
            -> Krida Khanda

        Ganesh_Puran_Upasana_Khand
            -> Upasana Khanda

        Mudgal_Puran_Khand_8
            -> Khanda 8
    """

    normalized = book_name.lower()

    # Ganesh Purana
    if "krida" in normalized:
        return "Krida Khanda"

    if "upasana" in normalized:
        return "Upasana Khanda"

    # Mudgal Purana
    match = re.search(
        r"mudgal_puran_khand_(\d+)",
        normalized
    )

    if match:
        return f"Khanda {match.group(1)}"

    return "Unknown"


# ============================================================
# 8. EXTRACT CHAPTER NUMBER
# ============================================================

def extract_chapter_number(chapter_info: str):
    """
    Extract numerical chapter number.

    Example:
        'Chapter 41 : The Great Story'
            -> 41
    """

    if not chapter_info:
        return None

    match = re.search(
        r"chapter\s+(\d+)",
        chapter_info,
        re.IGNORECASE
    )

    if match:
        return int(match.group(1))

    return None


# ============================================================
# 9. EXTRACT CHAPTER TITLE
# ============================================================

def extract_chapter_title(chapter_info: str) -> str:
    """
    Extract chapter title.

    Example:
        Chapter 41 : The Great Story

    Returns:
        The Great Story
    """

    if not chapter_info:
        return ""

    parts = chapter_info.split(":", 1)

    if len(parts) == 2:
        return parts[1].strip()

    return chapter_info.strip()


# ============================================================
# 10. CLEAN TEXT
# ============================================================

def clean_text(text: str) -> str:
    """
    Perform lightweight text normalization.

    IMPORTANT:
    We deliberately do not aggressively modify the scripture.
    The goal is only to remove obvious formatting noise.
    """

    if not text:
        return ""

    # Normalize Windows line endings
    text = text.replace("\r\n", "\n")

    # Normalize old Mac line endings
    text = text.replace("\r", "\n")

    # Remove excessive spaces
    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    # Reduce excessive blank lines
    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    return text.strip()


# ============================================================
# 11. ENRICH DOCUMENT METADATA
# ============================================================

def enrich_document_metadata(documents):
    """
    Add structured metadata to every chapter document.

    Existing metadata:
        book
        chapter_info

    New metadata:
        source_type
        source
        section
        chapter
        chapter_number
        chapter_title
        authority
        tradition
        language
        embedding_model
        embedding_version
        chunking_version
    """

    enriched_documents = []

    for doc in documents:

        original_metadata = dict(doc.metadata)

        book_name = original_metadata.get(
            "book",
            "Unknown"
        )

        chapter_info = original_metadata.get(
            "chapter_info",
            ""
        )

        source = identify_purana(book_name)

        section = extract_section(book_name)

        chapter_number = extract_chapter_number(
            chapter_info
        )

        chapter_title = extract_chapter_title(
            chapter_info
        )

        # Clean document content
        cleaned_content = clean_text(
            doc.page_content
        )

        # ----------------------------------------------------
        # Updated metadata
        # ----------------------------------------------------

        metadata = {

            # High-level classification
            "source_type": SOURCE_TYPE,

            "source": source,

            "authority": AUTHORITY,

            "tradition": TRADITION,

            "language": LANGUAGE,

            # Scriptural hierarchy
            "section": section,

            "chapter": chapter_info,

            "chapter_number": chapter_number,

            "chapter_title": chapter_title,

            # Preserve original loader information
            "original_book": book_name,

            "original_chapter_info": chapter_info,

            # Embedding information
            "embedding_model": EMBEDDING_MODEL_NAME,

            "embedding_version": EMBEDDING_VERSION,

            # Chunking information
            "chunk_size": CHUNK_SIZE,

            "chunk_overlap": CHUNK_OVERLAP,

            "chunking_version": CHUNKING_VERSION,
        }

        # Create a new LangChain Document
        from langchain_core.documents import Document

        enriched_doc = Document(
            page_content=cleaned_content,
            metadata=metadata
        )

        enriched_documents.append(
            enriched_doc
        )

    return enriched_documents


# ============================================================
# 12. SPLIT DOCUMENTS
# ============================================================

def split_documents(documents):
    """
    Split chapter-level documents into smaller chunks.

    Same configuration as the original notebook:
        chunk_size = 1000
        chunk_overlap = 200
    """

    print("\n" + "=" * 70)
    print("CHUNKING DOCUMENTS")
    print("=" * 70)

    text_splitter = RecursiveCharacterTextSplitter(

        chunk_size=CHUNK_SIZE,

        chunk_overlap=CHUNK_OVERLAP,

        separators=[
            "\n\n",
            "\n",
            " ",
            ""
        ]
    )

    chunks = text_splitter.split_documents(
        documents
    )

    print(
        f"\nSplit {len(documents)} chapters "
        f"into {len(chunks)} chunks."
    )

    if chunks:

        print("\nSample chunk:")
        print("-" * 70)

        print(
            chunks[0].page_content[:500]
        )

        print("\nMetadata:")
        print(chunks[0].metadata)

    return chunks


# ============================================================
# 13. CREATE DETERMINISTIC CHUNK ID
# ============================================================

def create_chunk_id(doc, chunk_index: int) -> str:
    """
    Create a deterministic ID for every chunk.

    Instead of:
        UUID4

    we generate IDs based on:
        source
        section
        chapter
        chunk content

    This makes ingestion reproducible.
    """

    source = doc.metadata.get(
        "source",
        "unknown"
    )

    section = doc.metadata.get(
        "section",
        "unknown"
    )

    chapter_number = doc.metadata.get(
        "chapter_number",
        "unknown"
    )

    content_hash = hashlib.sha256(
        doc.page_content.encode("utf-8")
    ).hexdigest()[:12]

    # Normalize strings
    source_slug = re.sub(
        r"[^a-z0-9]+",
        "_",
        source.lower()
    ).strip("_")

    section_slug = re.sub(
        r"[^a-z0-9]+",
        "_",
        section.lower()
    ).strip("_")

    return (
        f"{source_slug}_"
        f"{section_slug}_"
        f"ch_{chapter_number}_"
        f"chunk_{chunk_index:05d}_"
        f"{content_hash}"
    )


# ============================================================
# 14. ADD CHUNK-SPECIFIC METADATA
# ============================================================

def prepare_chunks(chunks):
    """
    Add chunk-level metadata and deterministic IDs.
    """

    print("\n" + "=" * 70)
    print("PREPARING CHUNKS")
    print("=" * 70)

    ids = []

    # Keep track of chunk number within each chapter
    chapter_counters = {}

    for chunk in chunks:

        source = chunk.metadata.get(
            "source",
            "unknown"
        )

        section = chunk.metadata.get(
            "section",
            "unknown"
        )

        chapter_number = chunk.metadata.get(
            "chapter_number",
            "unknown"
        )

        chapter_key = (
            source,
            section,
            chapter_number
        )

        if chapter_key not in chapter_counters:
            chapter_counters[chapter_key] = 0

        chunk_index = chapter_counters[
            chapter_key
        ]

        chapter_counters[
            chapter_key
        ] += 1

        # Create deterministic ID
        chunk_id = create_chunk_id(
            chunk,
            chunk_index
        )

        ids.append(chunk_id)

        # Add chunk metadata
        chunk.metadata.update({

            "chunk_id": chunk_id,

            "chunk_index": chunk_index,

            "chunk_length": len(
                chunk.page_content
            ),

            "chunk_size": CHUNK_SIZE,

            "chunk_overlap": CHUNK_OVERLAP,

            "embedding_model": EMBEDDING_MODEL_NAME,

            "embedding_version": EMBEDDING_VERSION,

            "chunking_version": CHUNKING_VERSION,

        })

    print(
        f"Prepared {len(chunks)} chunks."
    )

    print(
        f"Generated {len(ids)} deterministic IDs."
    )

    print("\nExample ID:")
    if ids:
        print(ids[0])

    return chunks, ids


# ============================================================
# 15. CREATE EMBEDDING MODEL
# ============================================================

def create_embedder():
    """
    Create the SentenceTransformer embedding model.

    This preserves the model used by the original notebook.
    """

    print("\n" + "=" * 70)
    print("LOADING EMBEDDING MODEL")
    print("=" * 70)

    print(
        f"Model: {EMBEDDING_MODEL_NAME}"
    )

    embedder = SentenceTransformerEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        device=None
    )

    print("Embedding model loaded.")

    return embedder


# ============================================================
# 16. CREATE ASTRA VECTOR STORE
# ============================================================

def create_vector_store(embedder):
    """
    Create Astra DB vector store.

    Collection:
        puranas
    """

    print("\n" + "=" * 70)
    print("CONNECTING TO ASTRA VECTOR STORE")
    print("=" * 70)

    vector_store = AstraDBVectorStore(

        collection_name=ASTRA_COLLECTION_NAME,

        embedding=embedder,

        token=ASTRA_DB_APPLICATION_TOKEN,

        api_endpoint=ASTRA_DB_API_ENDPOINT,
    )

    print(
        f"Collection: {ASTRA_COLLECTION_NAME}"
    )

    return vector_store


# ============================================================
# 17. INSERT INTO ASTRA DB
# ============================================================

def insert_chunks(
    vector_store,
    chunks,
    ids
):
    """
    Insert chunks into Astra DB.

    Uses deterministic IDs so that the same chunk can be
    identified consistently between ingestion runs.
    """

    print("\n" + "=" * 70)
    print("INSERTING INTO ASTRA DB")
    print("=" * 70)

    texts = [
        chunk.page_content
        for chunk in chunks
    ]

    metadatas = [
        chunk.metadata
        for chunk in chunks
    ]

    print(
        f"Uploading {len(texts)} chunks..."
    )

    document_ids = vector_store.add_texts(

        texts=texts,

        metadatas=metadatas,

        ids=ids
    )

    print(
        f"\nSuccessfully inserted "
        f"{len(document_ids)} chunks."
    )

    if document_ids:

        print("\nFirst 3 IDs:")

        for document_id in document_ids[:3]:
            print(
                f"  {document_id}"
            )

    return document_ids


# ============================================================
# 18. PRINT INGESTION SUMMARY
# ============================================================

def print_summary(
    documents,
    chunks,
    document_ids
):
    """
    Print final ingestion statistics.
    """

    print("\n")
    print("=" * 70)
    print("INGESTION COMPLETE")
    print("=" * 70)

    print(
        f"Database          : {ASTRA_DB_NAME}"
    )

    print(
        f"Keyspace          : {ASTRA_DB_KEYSPACE}"
    )

    print(
        f"Collection        : {ASTRA_COLLECTION_NAME}"
    )

    print(
        f"Source Type       : {SOURCE_TYPE}"
    )

    print(
        f"Authority         : {AUTHORITY}"
    )

    print(
        f"Embedding Model   : {EMBEDDING_MODEL_NAME}"
    )

    print(
        f"Embedding Version : {EMBEDDING_VERSION}"
    )

    print(
        f"Chunking Version  : {CHUNKING_VERSION}"
    )

    print(
        f"Original Chapters : {len(documents)}"
    )

    print(
        f"Total Chunks      : {len(chunks)}"
    )

    print(
        f"Uploaded          : {len(document_ids)}"
    )

    print("=" * 70)


# ============================================================
# 19. MAIN PIPELINE
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("GANESHRAG - PURANA INGESTION PIPELINE")
    print("=" * 70)

    # --------------------------------------------------------
    # Step 1: Validate environment
    # --------------------------------------------------------

    validate_environment()

    # --------------------------------------------------------
    # Step 2: Load chapters
    # --------------------------------------------------------

    documents = load_puranas()

    # --------------------------------------------------------
    # Step 3: Enrich metadata
    # --------------------------------------------------------

    documents = enrich_document_metadata(
        documents
    )

    print("\nMetadata enrichment complete.")

    if documents:

        print("\nExample enriched metadata:")
        print("-" * 70)

        for key, value in documents[0].metadata.items():
            print(
                f"{key}: {value}"
            )

    # --------------------------------------------------------
    # Step 4: Split chapters into chunks
    # --------------------------------------------------------

    chunks = split_documents(
        documents
    )

    # --------------------------------------------------------
    # Step 5: Prepare deterministic IDs
    # --------------------------------------------------------

    chunks, ids = prepare_chunks(
        chunks
    )

    # --------------------------------------------------------
    # Step 6: Load embedding model
    # --------------------------------------------------------

    embedder = create_embedder()

    # --------------------------------------------------------
    # Step 7: Create Astra vector store
    # --------------------------------------------------------

    vector_store = create_vector_store(
        embedder
    )

    # --------------------------------------------------------
    # Step 8: Upload chunks
    # --------------------------------------------------------

    document_ids = insert_chunks(
        vector_store,
        chunks,
        ids
    )

    # --------------------------------------------------------
    # Step 9: Summary
    # --------------------------------------------------------

    print_summary(
        documents,
        chunks,
        document_ids
    )


# ============================================================
# 20. SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()