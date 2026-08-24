"""
============================================================
GaneshRAG - Research / ETIC Ingestion Pipeline
============================================================

Purpose:
    Ingest scholarly / research books from:

        Model/docs/etic/

    into Astra DB.

Input format:

    Book text...

    --- Page 2 ---

    Page 2 text...

    --- Page 3 ---

    Page 3 text...

Each page is first separated into a document and then
split into smaller chunks for vector retrieval.

Astra DB:

    Database:
        ganesa_data

    Keyspace:
        default_keyspace

    Collection:
        research

Embedding:

    all-MiniLM-L6-v2

Metadata philosophy:

    Research books are treated as:

        source_type = research
        authority   = secondary_scholarship

They are NOT treated as primary scripture.

============================================================
"""


# ============================================================
# 1. IMPORTS
# ============================================================

import os
import re
import hashlib
from pathlib import Path

import dotenv

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_astradb import AstraDBVectorStore

from embedder import SentenceTransformerEmbeddings


# ============================================================
# 2. ENVIRONMENT
# ============================================================

dotenv.load_dotenv(override=True)


# ============================================================
# 3. PROJECT PATHS
# ============================================================

# ingest_research.py:
#
# Model/
# └── ingestion/
#     └── ingest_research.py
#
# Therefore:
#
# .parent       -> ingestion/
# .parent.parent -> Model/

BASE_DIR = Path(
    __file__
).resolve().parent.parent


# Research documents:
#
# Model/docs/etic/

DOCS_DIR = BASE_DIR / "docs"

RESEARCH_DIR = DOCS_DIR / "etic"


# ============================================================
# 4. ASTRA DB CONFIGURATION
# ============================================================

ASTRA_DB_API_ENDPOINT = os.getenv(
    "ASTRA_DB_API_ENDPOINT"
)

ASTRA_DB_APPLICATION_TOKEN = os.getenv(
    "ASTRA_DB_APPLICATION_TOKEN"
)

ASTRA_DB_NAME = os.getenv(
    "ASTRA_DB_NAME",
    "ganesa_data"
)

ASTRA_DB_KEYSPACE = os.getenv(
    "ASTRA_DB_KEYSPACE",
    "default_keyspace"
)

# IMPORTANT:
# This is a separate collection from:
#
# puranas
# upanishads
# sahastranaam
# iconography

ASTRA_COLLECTION_NAME = "research"


# ============================================================
# 5. EMBEDDING CONFIGURATION
# ============================================================

EMBEDDING_MODEL_NAME = (
    "all-MiniLM-L6-v2"
)

EMBEDDING_VERSION = "v1"


# ============================================================
# 6. CHUNKING CONFIGURATION
# ============================================================

CHUNK_SIZE = 1000

CHUNK_OVERLAP = 200

CHUNKING_VERSION = "v1"


# ============================================================
# 7. SOURCE METADATA
# ============================================================

SOURCE_TYPE = "research"

AUTHORITY = "secondary_scholarship"

LANGUAGE = "English"

# Research books are not automatically assumed to belong
# to a particular religious tradition.

TRADITION = "Hindu Studies"


# ============================================================
# 8. VALIDATE ENVIRONMENT
# ============================================================

def validate_environment():
    """
    Validate environment variables and research directory.
    """

    print("\n" + "=" * 70)
    print("VALIDATING ENVIRONMENT")
    print("=" * 70)

    # --------------------------------------------------------
    # Astra credentials
    # --------------------------------------------------------

    if not ASTRA_DB_API_ENDPOINT:

        raise RuntimeError(
            "ASTRA_DB_API_ENDPOINT is not defined "
            "in the .env file."
        )

    if not ASTRA_DB_APPLICATION_TOKEN:

        raise RuntimeError(
            "ASTRA_DB_APPLICATION_TOKEN is not defined "
            "in the .env file."
        )

    # --------------------------------------------------------
    # Research directory
    # --------------------------------------------------------

    if not RESEARCH_DIR.exists():

        raise FileNotFoundError(
            f"Research directory not found:\n"
            f"{RESEARCH_DIR}"
        )

    if not RESEARCH_DIR.is_dir():

        raise NotADirectoryError(
            f"Research path is not a directory:\n"
            f"{RESEARCH_DIR}"
        )

    print(
        f"Project root : {BASE_DIR}"
    )

    print(
        f"Research dir : {RESEARCH_DIR}"
    )

    print(
        "Environment validation successful."
    )


# ============================================================
# 9. GET RESEARCH FILES
# ============================================================

def get_research_files():
    """
    Find research books inside docs/etic/.

    Supported formats:

        .txt
        .md
        .text

    If your books are stored as PDFs, this script should be
    extended with a PDF extraction stage rather than treating
    PDF bytes as text.
    """

    supported_extensions = {
        ".txt",
        ".md",
        ".text"
    }

    files = []

    for path in RESEARCH_DIR.iterdir():

        if not path.is_file():
            continue

        if path.suffix.lower() in supported_extensions:

            files.append(path)

    files.sort(
        key=lambda x: x.name.lower()
    )

    if not files:

        raise FileNotFoundError(
            f"No supported research files found in:\n"
            f"{RESEARCH_DIR}\n\n"
            f"Supported extensions: "
            f"{', '.join(sorted(supported_extensions))}"
        )

    print(
        f"\nFound {len(files)} research book(s):"
    )

    for file in files:

        print(
            f"  - {file.name}"
        )

    return files


# ============================================================
# 10. EXTRACT BOOK TITLE
# ============================================================

def extract_book_title(file_path: Path) -> str:
    """
    Convert filename into a human-readable book title.

    Examples:

        ganesha_and_his_cult.txt
            ->
        Ganesha And His Cult

    If you want exact titles, you can later add a
    book_metadata.json configuration file.
    """

    title = file_path.stem

    # Replace separators
    title = title.replace(
        "_",
        " "
    )

    title = title.replace(
        "-",
        " "
    )

    # Normalize whitespace
    title = re.sub(
        r"\s+",
        " ",
        title
    ).strip()

    # Title case
    title = title.title()

    return title


# ============================================================
# 11. READ FILE
# ============================================================

def read_book(file_path: Path) -> str:
    """
    Read a research book as UTF-8 text.
    """

    try:

        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as file:

            return file.read()

    except UnicodeDecodeError:

        # Fallback for files that may have been saved using
        # another common encoding.

        with open(
            file_path,
            "r",
            encoding="utf-8-sig"
        ) as file:

            return file.read()


# ============================================================
# 12. CLEAN PAGE MARKERS
# ============================================================

def normalize_page_markers(text: str) -> str:
    """
    Normalize page marker formatting.

    Expected:

        --- Page 2 ---

    Also accepts variants such as:

        ---Page 2---

        --- page 2 ---

        -- Page 2 --

    The parser is intentionally tolerant.
    """

    text = text.replace(
        "\r\n",
        "\n"
    )

    text = text.replace(
        "\r",
        "\n"
    )

    # Normalize page markers to:
    #
    # --- Page N ---

    pattern = re.compile(
        r"-{2,}\s*"
        r"page\s+(\d+)"
        r"\s*-{2,}",
        re.IGNORECASE
    )

    text = pattern.sub(
        lambda match:
            f"\n--- Page {match.group(1)} ---\n",
        text
    )

    return text


# ============================================================
# 13. SPLIT BOOK INTO PAGES
# ============================================================

def split_book_into_pages(
    text: str,
    file_path: Path
):
    """
    Split a book using:

        --- Page N ---

    markers.

    Returns:
        List[Document]
    """

    text = normalize_page_markers(
        text
    )

    # --------------------------------------------------------
    # Regex for page marker
    # --------------------------------------------------------

    page_pattern = re.compile(
        r"---\s*Page\s+(\d+)\s*---",
        re.IGNORECASE
    )

    matches = list(
        page_pattern.finditer(text)
    )

    documents = []

    book_title = extract_book_title(
        file_path
    )

    # --------------------------------------------------------
    # Case 1:
    # No page markers
    # --------------------------------------------------------

    if not matches:

        cleaned_text = clean_text(
            text
        )

        if cleaned_text:

            documents.append(
                Document(

                    page_content=
                        cleaned_text,

                    metadata={

                        "source_type":
                            SOURCE_TYPE,

                        "authority":
                            AUTHORITY,

                        "source":
                            book_title,

                        "book_title":
                            book_title,

                        "file_name":
                            file_path.name,

                        "page_number":
                            None,

                        "page_label":
                            "Unknown",

                        "section":
                            None,

                        "chapter":
                            None,

                        "chapter_number":
                            None,

                        "chapter_title":
                            None,

                        "language":
                            LANGUAGE,

                        "tradition":
                            TRADITION,

                        "original_file":
                            str(file_path.relative_to(
                                BASE_DIR
                            ))
                    }
                )
            )

        return documents

    # --------------------------------------------------------
    # Content before first page marker
    # --------------------------------------------------------

    preamble = text[
        :matches[0].start()
    ].strip()

    if preamble:

        preamble = clean_text(
            preamble
        )

        if preamble:

            documents.append(
                Document(

                    page_content=
                        preamble,

                    metadata={

                        "source_type":
                            SOURCE_TYPE,

                        "authority":
                            AUTHORITY,

                        "source":
                            book_title,

                        "book_title":
                            book_title,

                        "file_name":
                            file_path.name,

                        "page_number":
                            None,

                        "page_label":
                            "Preamble",

                        "section":
                            None,

                        "chapter":
                            None,

                        "chapter_number":
                            None,

                        "chapter_title":
                            None,

                        "language":
                            LANGUAGE,

                        "tradition":
                            TRADITION,

                        "original_file":
                            str(file_path.relative_to(
                                BASE_DIR
                            ))
                    }
                )
            )

    # --------------------------------------------------------
    # Extract pages
    # --------------------------------------------------------

    for index, match in enumerate(
        matches
    ):

        page_number = int(
            match.group(1)
        )

        # Start after:
        #
        # --- Page N ---

        content_start = match.end()

        # End before next marker
        # or end of document.

        if index + 1 < len(matches):

            content_end = (
                matches[index + 1].start()
            )

        else:

            content_end = len(text)

        page_content = text[
            content_start:
            content_end
        ]

        page_content = clean_text(
            page_content
        )

        # Ignore completely empty pages
        if not page_content:
            continue

        metadata = {

            # ------------------------------------------------
            # Source classification
            # ------------------------------------------------

            "source_type":
                SOURCE_TYPE,

            "authority":
                AUTHORITY,

            # ------------------------------------------------
            # Source information
            # ------------------------------------------------

            "source":
                book_title,

            "book_title":
                book_title,

            "file_name":
                file_path.name,

            "original_file":
                str(
                    file_path.relative_to(
                        BASE_DIR
                    )
                ),

            # ------------------------------------------------
            # Page information
            # ------------------------------------------------

            "page_number":
                page_number,

            "page_label":
                f"Page {page_number}",

            # ------------------------------------------------
            # Structural metadata
            #
            # These are deliberately None because a page
            # marker alone does not establish a chapter or
            # section.
            # ------------------------------------------------

            "section":
                None,

            "chapter":
                None,

            "chapter_number":
                None,

            "chapter_title":
                None,

            # ------------------------------------------------
            # Language / tradition
            # ------------------------------------------------

            "language":
                LANGUAGE,

            "tradition":
                TRADITION,

            # ------------------------------------------------
            # Version information
            # ------------------------------------------------

            "embedding_model":
                EMBEDDING_MODEL_NAME,

            "embedding_version":
                EMBEDDING_VERSION,

            "chunking_version":
                CHUNKING_VERSION
        }

        documents.append(
            Document(

                page_content=
                    page_content,

                metadata=
                    metadata
            )
        )

    return documents


# ============================================================
# 14. CLEAN TEXT
# ============================================================

def clean_text(text: str) -> str:
    """
    Perform conservative text cleaning.

    We deliberately avoid aggressive modifications because
    these are scholarly texts and formatting may carry meaning.
    """

    if not text:

        return ""

    # Normalize line endings

    text = text.replace(
        "\r\n",
        "\n"
    )

    text = text.replace(
        "\r",
        "\n"
    )

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
# 15. LOAD ALL RESEARCH BOOKS
# ============================================================

def load_research_books():
    """
    Load every research book and split it into pages.
    """

    print("\n" + "=" * 70)
    print("LOADING RESEARCH BOOKS")
    print("=" * 70)

    files = get_research_files()

    all_documents = []

    for file_path in files:

        print(
            f"\nReading: {file_path.name}"
        )

        text = read_book(
            file_path
        )

        pages = split_book_into_pages(
            text,
            file_path
        )

        print(
            f"  Pages loaded: {len(pages)}"
        )

        all_documents.extend(
            pages
        )

    print(
        f"\nTotal page documents loaded: "
        f"{len(all_documents)}"
    )

    return all_documents


# ============================================================
# 16. SPLIT PAGES INTO CHUNKS
# ============================================================

def split_documents_into_chunks(
    documents
):
    """
    Split each page into smaller chunks.

    IMPORTANT:
    The splitter operates on each page document independently,
    preserving the page number in the metadata.
    """

    print("\n" + "=" * 70)
    print("CHUNKING RESEARCH PAGES")
    print("=" * 70)

    text_splitter = (
        RecursiveCharacterTextSplitter(

            chunk_size=
                CHUNK_SIZE,

            chunk_overlap=
                CHUNK_OVERLAP,

            separators=[
                "\n\n",
                "\n",
                " ",
                ""
            ]
        )
    )

    chunks = (
        text_splitter
        .split_documents(
            documents
        )
    )

    print(
        f"Pages : {len(documents)}"
    )

    print(
        f"Chunks: {len(chunks)}"
    )

    return chunks


# ============================================================
# 17. CREATE DETERMINISTIC CHUNK ID
# ============================================================

def create_chunk_id(
    document,
    chunk_index
):
    """
    Create deterministic chunk IDs.

    Example:

        research_ganesha_and_his_cult_page_42_chunk_00003_a82f...
    """

    metadata = document.metadata

    source = metadata.get(
        "book_title",
        "unknown"
    )

    page_number = metadata.get(
        "page_number",
        "unknown"
    )

    content_hash = hashlib.sha256(
        document.page_content.encode(
            "utf-8"
        )
    ).hexdigest()[:12]

    # Create safe slug

    source_slug = re.sub(
        r"[^a-zA-Z0-9]+",
        "_",
        source.lower()
    ).strip("_")

    return (
        f"research_"
        f"{source_slug}_"
        f"page_{page_number}_"
        f"chunk_{chunk_index:05d}_"
        f"{content_hash}"
    )


# ============================================================
# 18. ADD CHUNK METADATA
# ============================================================

def prepare_chunks(
    chunks
):
    """
    Add chunk-specific metadata.

    Chunk index resets for every page.
    """

    print("\n" + "=" * 70)
    print("PREPARING CHUNKS")
    print("=" * 70)

    counters = {}

    ids = []

    for chunk in chunks:

        metadata = chunk.metadata

        book_title = metadata.get(
            "book_title",
            "unknown"
        )

        page_number = metadata.get(
            "page_number",
            "unknown"
        )

        page_key = (
            book_title,
            page_number
        )

        # Initialize page counter

        if page_key not in counters:

            counters[page_key] = 0

        chunk_index = counters[
            page_key
        ]

        counters[
            page_key
        ] += 1

        # ----------------------------------------------------
        # Create deterministic ID
        # ----------------------------------------------------

        chunk_id = create_chunk_id(
            chunk,
            chunk_index
        )

        ids.append(
            chunk_id
        )

        # ----------------------------------------------------
        # Update metadata
        # ----------------------------------------------------

        chunk.metadata.update({

            "chunk_id":
                chunk_id,

            "chunk_index":
                chunk_index,

            "chunk_length":
                len(
                    chunk.page_content
                ),

            "chunk_size":
                CHUNK_SIZE,

            "chunk_overlap":
                CHUNK_OVERLAP,

            "embedding_model":
                EMBEDDING_MODEL_NAME,

            "embedding_version":
                EMBEDDING_VERSION,

            "chunking_version":
                CHUNKING_VERSION,

            "citation":
                create_citation(
                    chunk.metadata
                )
        })

    print(
        f"Prepared {len(chunks)} chunks."
    )

    return chunks, ids


# ============================================================
# 19. CREATE CITATION
# ============================================================

def create_citation(
    metadata
):
    """
    Generate a citation string from metadata.

    Example:

        (Ganesha And His Cult, p. 42)

    If page number is unavailable:

        (Ganesha And His Cult)
    """

    book_title = metadata.get(
        "book_title",
        "Unknown Source"
    )

    page_number = metadata.get(
        "page_number"
    )

    if page_number is not None:

        return (
            f"({book_title}, "
            f"p. {page_number})"
        )

    return (
        f"({book_title})"
    )


# ============================================================
# 20. CREATE EMBEDDING MODEL
# ============================================================

def create_embedder():
    """
    Load the same embedding model used by the rest of
    the GaneshRAG pipeline.
    """

    print("\n" + "=" * 70)
    print("LOADING EMBEDDING MODEL")
    print("=" * 70)

    print(
        f"Model: {EMBEDDING_MODEL_NAME}"
    )

    embedder = (
        SentenceTransformerEmbeddings(

            model_name=
                EMBEDDING_MODEL_NAME,

            device=None
        )
    )

    print(
        "Embedding model loaded."
    )

    return embedder


# ============================================================
# 21. CREATE ASTRA VECTOR STORE
# ============================================================

def create_vector_store(
    embedder
):
    """
    Connect to the research collection.

    Collection:

        research
    """

    print("\n" + "=" * 70)
    print("CONNECTING TO ASTRA VECTOR STORE")
    print("=" * 70)

    vector_store = (
        AstraDBVectorStore(

            collection_name=
                ASTRA_COLLECTION_NAME,

            embedding=
                embedder,

            token=
                ASTRA_DB_APPLICATION_TOKEN,

            api_endpoint=
                ASTRA_DB_API_ENDPOINT
        )
    )

    print(
        f"Database   : {ASTRA_DB_NAME}"
    )

    print(
        f"Keyspace   : {ASTRA_DB_KEYSPACE}"
    )

    print(
        f"Collection : {ASTRA_COLLECTION_NAME}"
    )

    return vector_store


# ============================================================
# 22. INSERT INTO ASTRA DB
# ============================================================

def insert_chunks(
    vector_store,
    chunks,
    ids
):
    """
    Upload chunks and metadata to Astra DB.
    """

    print("\n" + "=" * 70)
    print("UPLOADING RESEARCH CHUNKS")
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

    document_ids = (
        vector_store.add_texts(

            texts=
                texts,

            metadatas=
                metadatas,

            ids=
                ids
        )
    )

    print(
        f"\nSuccessfully uploaded "
        f"{len(document_ids)} chunks."
    )

    return document_ids


# ============================================================
# 23. PRINT SAMPLE METADATA
# ============================================================

def print_sample_metadata(
    chunks
):
    """
    Print metadata from the first chunk for verification.
    """

    if not chunks:
        return

    print("\n" + "=" * 70)
    print("SAMPLE CHUNK")
    print("=" * 70)

    chunk = chunks[0]

    print("\nContent:")
    print(
        chunk.page_content[:500]
    )

    print("\nMetadata:")

    for key, value in (
        chunk.metadata.items()
    ):

        print(
            f"{key}: {value}"
        )


# ============================================================
# 24. PRINT SUMMARY
# ============================================================

def print_summary(
    page_documents,
    chunks,
    document_ids
):
    """
    Print final ingestion statistics.
    """

    print("\n")
    print("=" * 70)
    print("RESEARCH INGESTION COMPLETE")
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
        f"Pages             : {len(page_documents)}"
    )

    print(
        f"Chunks            : {len(chunks)}"
    )

    print(
        f"Uploaded          : {len(document_ids)}"
    )

    print("=" * 70)


# ============================================================
# 25. MAIN
# ============================================================

def main():

    print("\n")

    print("=" * 70)

    print(
        "GANESHRAG - RESEARCH INGESTION PIPELINE"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # Step 1
    # --------------------------------------------------------

    validate_environment()

    # --------------------------------------------------------
    # Step 2
    # --------------------------------------------------------

    page_documents = (
        load_research_books()
    )

    if not page_documents:

        raise RuntimeError(
            "No research pages were loaded."
        )

    # --------------------------------------------------------
    # Step 3
    # --------------------------------------------------------

    chunks = (
        split_documents_into_chunks(
            page_documents
        )
    )

    if not chunks:

        raise RuntimeError(
            "No chunks were generated."
        )

    # --------------------------------------------------------
    # Step 4
    # --------------------------------------------------------

    chunks, ids = (
        prepare_chunks(
            chunks
        )
    )

    # --------------------------------------------------------
    # Step 5
    # --------------------------------------------------------

    print_sample_metadata(
        chunks
    )

    # --------------------------------------------------------
    # Step 6
    # --------------------------------------------------------

    embedder = (
        create_embedder()
    )

    # --------------------------------------------------------
    # Step 7
    # --------------------------------------------------------

    vector_store = (
        create_vector_store(
            embedder
        )
    )

    # --------------------------------------------------------
    # Step 8
    # --------------------------------------------------------

    document_ids = (
        insert_chunks(

            vector_store,

            chunks,

            ids
        )
    )

    # --------------------------------------------------------
    # Step 9
    # --------------------------------------------------------

    print_summary(

        page_documents,

        chunks,

        document_ids
    )


# ============================================================
# 26. ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()