from pathlib import Path
import hashlib
import os
import re

from langchain_core.documents import Document
from langchain_astradb import AstraDBVectorStore

from embedder import SentenceTransformerEmbeddings
import dotenv


# ============================================================
# CONFIGURATION
# ============================================================
dotenv.load_dotenv(override=True)

BASE_DIR = Path(__file__).resolve().parent.parent

DOCS_DIR = BASE_DIR / "docs" / "upanishads"

COLLECTION_NAME = "upanishads"


# ============================================================
# ASTRA DB CONFIGURATION
# ============================================================

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

SOURCE_TYPE = "upanishad"

DOCUMENT_TYPE = "upanishadic_commentary"

AUTHORITY = "traditional_scripture"

TRADITION = "Ganapatya"

UPANISHAD_INFO = {
    "ganesh_atharvashirsha_upanishad.txt": {
        "name": "Ganesh Atharvashirsha Upanishad",
        "short_name": "Ganesh Atharvashirsha",
        "author": "Balvinayak Lalsare Maharaj",
    },

    "ganesha_tapini.txt": {
        "name": "Ganesh Tapini Upanishad",
        "short_name": "Ganesh Tapini",
        "author": "Sundar Hattangadi",
    },

    "herambha_upanishad.txt": {
        "name": "Herambha Upanishad",
        "short_name": "Herambha Upanishad",
        "author": "K. Koushik",
    },

    "vallabhesha_upanishad.txt": {
        "name": "Vallabhesha Upanishad",
        "short_name": "Vallabhesha Upanishad",
        "author": "Gudrun Buhnemann",
    },
}


# ============================================================
# FILE ORDER
# ============================================================

FILE_ORDER = {
    "ganesh_atharvashirsha_upanishad.txt": 1,
    "ganesh_tapini.txt": 2,
    "herambha_upanishad.txt": 3,
    "vallabhesha_upanishad.txt": 4,
}


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text: str) -> str:
    """
    Conservative text normalization.

    Does not translate, rewrite, or remove Sanskrit terminology.
    """

    # Normalize line endings
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Remove trailing whitespace
    text = "\n".join(
        line.rstrip()
        for line in text.splitlines()
    )

    # Remove excessive blank lines
    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    return text.strip()


# ============================================================
# MANTRA NUMBER DETECTION
# ============================================================

def extract_mantra_number(text: str):
    """
    Attempt to identify the mantra/verse number.

    Supports common forms such as:

        1.
        1)
        ॥ १ ॥
        ॥ 2 ॥
        ॥ २॥
        Mantra 2
        Verse 2
        Shloka 2

    Returns an integer where confidently identifiable.
    """

    # --------------------------------------------------------
    # Devanagari numerals inside verse ending
    # Example:
    # ॥ २॥
    # --------------------------------------------------------

    devanagari_digits = "०१२३४५६७८९"

    match = re.search(
        r"॥\s*([०-९]+)\s*॥",
        text,
    )

    if match:

        number_string = match.group(1)

        translation = str.maketrans(
            devanagari_digits,
            "0123456789",
        )

        try:
            return int(
                number_string.translate(
                    translation
                )
            )
        except ValueError:
            pass

    # --------------------------------------------------------
    # English labels
    # --------------------------------------------------------

    match = re.search(
        r"\b(?:Mantra|Verse|Shloka)\s+(\d+)\b",
        text,
        re.IGNORECASE,
    )

    if match:
        return int(
            match.group(1)
        )

    # --------------------------------------------------------
    # Simple numbered heading
    # --------------------------------------------------------

    match = re.match(
        r"^\s*(\d+)[.)]\s+",
        text,
    )

    if match:
        return int(
            match.group(1)
        )

    return None


# ============================================================
# MANTRA / VERSE SPLITTING
# ============================================================

def split_by_mantra(text: str):
    """
    Split an Upanishad into logical mantra-level sections.

    The function primarily looks for Devanagari verse endings
    such as:

        ॥ २॥

    This works particularly well for Herambha Upanishad.

    If no clear mantra boundaries are found, the whole file
    is returned as one logical section and can subsequently
    be split using paragraph-aware chunking.
    """

    # --------------------------------------------------------
    # Find verse endings
    # --------------------------------------------------------

    pattern = re.compile(
        r"॥\s*[०-९0-9]+\s*॥"
    )

    matches = list(
        pattern.finditer(text)
    )

    if not matches:
        return [
            {
                "mantra_number": None,
                "text": text.strip(),
            }
        ]

    sections = []

    start = 0

    for index, match in enumerate(matches):

        end = match.end()

        section_text = text[
            start:end
        ].strip()

        if section_text:

            mantra_number = extract_mantra_number(
                section_text
            )

            sections.append(
                {
                    "mantra_number": mantra_number,
                    "text": section_text,
                }
            )

        start = end

    # --------------------------------------------------------
    # Preserve any remaining text after the last mantra
    # --------------------------------------------------------

    remaining = text[
        start:
    ].strip()

    if remaining:

        sections.append(
            {
                "mantra_number": extract_mantra_number(
                    remaining
                ),
                "text": remaining,
            }
        )

    return sections


# ============================================================
# PARAGRAPH-AWARE FALLBACK
# ============================================================

def split_long_section(
    text: str,
    max_characters: int = 6000,
):
    """
    Split unusually long sections while preserving paragraph
    boundaries as much as possible.

    This is a fallback for English-only Upanishad files or
    unusually long commentary sections.
    """

    if len(text) <= max_characters:

        return [
            text.strip()
        ]

    paragraphs = [
        p.strip()
        for p in re.split(
            r"\n\s*\n",
            text,
        )
        if p.strip()
    ]

    chunks = []

    current = ""

    for paragraph in paragraphs:

        # If adding the next paragraph remains within the
        # preferred size, keep it together.
        if (
            not current
            or len(current) + len(paragraph) + 2
            <= max_characters
        ):

            if current:
                current += "\n\n"

            current += paragraph

        else:

            chunks.append(
                current.strip()
            )

            current = paragraph

    if current:
        chunks.append(
            current.strip()
        )

    return chunks


# ============================================================
# CHUNK ID
# ============================================================

def create_chunk_id(
    upanishad_name: str,
    mantra_number,
    chunk_index: int,
    content: str,
):
    """
    Generate a deterministic SHA-256 ID.
    """

    raw = (
        f"{upanishad_name}|"
        f"{mantra_number}|"
        f"{chunk_index}|"
        f"{content}"
    )

    return hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()


# ============================================================
# CITATION
# ============================================================

def create_citation(
    upanishad_name: str,
    mantra_number,
):
    """
    Generate a human-readable citation.
    """

    if mantra_number is not None:

        return (
            f"({upanishad_name}, "
            f"Mantra {mantra_number})"
        )

    return (
        f"({upanishad_name})"
    )


# ============================================================
# LOAD DOCUMENTS
# ============================================================

def load_documents():

    if not DOCS_DIR.exists():

        raise FileNotFoundError(
            f"Directory not found: {DOCS_DIR}"
        )

    files = list(
        DOCS_DIR.glob("*.txt")
    )

    if not files:

        raise FileNotFoundError(
            f"No .txt files found in: {DOCS_DIR}"
        )

    # Explicit ordering
    files.sort(
        key=lambda path: FILE_ORDER.get(
            path.name,
            999,
        )
    )

    print("=" * 70)
    print("UPANISHAD INGESTION")
    print("=" * 70)

    print(
        f"Source directory : {DOCS_DIR}"
    )

    print(
        f"Collection       : {COLLECTION_NAME}"
    )

    print(
        f"Files found      : {len(files)}"
    )

    print()

    documents = []

    # --------------------------------------------------------
    # Process each Upanishad
    # --------------------------------------------------------

    for file_path in files:

        filename = file_path.name

        info = UPANISHAD_INFO.get(
            filename
        )

        if not info:

            print(
                f"WARNING: No metadata configured "
                f"for {filename}"
            )

            continue

        upanishad_name = info["name"]
        short_name = info["short_name"]
        author = info["author"]

        print(
            f"Reading: {filename}"
        )

        print(
            f"  Upanishad : {upanishad_name}"
        )

        print(
            f"  Author    : {author}"
        )

        # ----------------------------------------------------
        # Read file
        # ----------------------------------------------------

        try:

            text = file_path.read_text(
                encoding="utf-8"
            )

        except UnicodeDecodeError:

            text = file_path.read_text(
                encoding="utf-8-sig"
            )

        text = clean_text(
            text
        )

        if not text:

            print(
                "  WARNING: Empty file — skipped."
            )

            continue

        # ----------------------------------------------------
        # Split into mantra-level sections
        # ----------------------------------------------------

        mantra_sections = split_by_mantra(
            text
        )

        retrieval_units = []

        # ----------------------------------------------------
        # Further split unusually long sections
        # ----------------------------------------------------

        for section in mantra_sections:

            mantra_number = section[
                "mantra_number"
            ]

            section_text = section[
                "text"
            ]

            subchunks = split_long_section(
                section_text
            )

            for subchunk_index, subchunk in enumerate(
                subchunks
            ):

                retrieval_units.append(
                    {
                        "mantra_number": mantra_number,
                        "text": subchunk,
                        "subchunk_index": subchunk_index,
                    }
                )

        total_units = len(
            retrieval_units
        )

        print(
            f"  Retrieval units: {total_units}"
        )

        # ----------------------------------------------------
        # Create LangChain Documents
        # ----------------------------------------------------

        for chunk_index, unit in enumerate(
            retrieval_units
        ):

            content = unit["text"]

            mantra_number = unit[
                "mantra_number"
            ]

            subchunk_index = unit[
                "subchunk_index"
            ]

            chunk_id = create_chunk_id(
                upanishad_name=upanishad_name,
                mantra_number=mantra_number,
                chunk_index=chunk_index,
                content=content,
            )

            citation = create_citation(
                upanishad_name=upanishad_name,
                mantra_number=mantra_number,
            )

            # ------------------------------------------------
            # Metadata
            # ------------------------------------------------

            metadata = {

                # ============================================
                # Collection
                # ============================================

                "collection": COLLECTION_NAME,

                # ============================================
                # Source classification
                # ============================================

                "source_type": SOURCE_TYPE,

                "document_type": DOCUMENT_TYPE,

                # ============================================
                # Upanishad
                # ============================================

                "upanishad": upanishad_name,

                "upanishad_short_name": short_name,

                "source": upanishad_name,

                "source_name": upanishad_name,

                "text_source": upanishad_name,

                # ============================================
                # Author
                # ============================================

                "author": author,

                # ============================================
                # Authority / tradition
                # ============================================

                "authority": AUTHORITY,

                "tradition": TRADITION,

                # ============================================
                # File information
                # ============================================

                "file_name": filename,

                "file_path": str(
                    file_path.relative_to(
                        BASE_DIR
                    )
                ),

                "document_title": upanishad_name,

                # ============================================
                # Structural information
                # ============================================

                "section": (
                    f"Mantra {mantra_number}"
                    if mantra_number is not None
                    else ""
                ),

                "mantra_number": (
                    mantra_number
                    if mantra_number is not None
                    else -1
                ),

                "subchunk_index": (
                    subchunk_index
                ),

                # ============================================
                # Chunk information
                # ============================================

                "chunk_id": chunk_id,

                "chunk_index": chunk_index,

                "total_chunks": total_units,

                # ============================================
                # Generic structural metadata
                # ============================================

                "chapter": "",

                "chapter_number": "",

                "chapter_title": "",

                "page_number": "",

                # ============================================
                # Citation
                # ============================================

                "citation": citation,
            }

            documents.append(
                Document(
                    page_content=content,
                    metadata=metadata,
                )
            )

    print()

    print(
        f"Total retrieval units prepared: "
        f"{len(documents)}"
    )

    return documents


# ============================================================
# EMBEDDINGS
# ============================================================

def create_embeddings():

    print()
    print("=" * 70)
    print("LOADING EMBEDDING MODEL")
    print("=" * 70)

    print(
        "Using SentenceTransformerEmbeddings "
        "from embedder.py"
    )

    embeddings = SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2",
        device=None,
    )

    print(
        "Embedding model loaded."
    )

    return embeddings


# ============================================================
# ASTRA VECTOR STORE
# ============================================================

def create_vector_store(
    embeddings,
):

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

    print(
        "Astra DB connection initialized."
    )

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

        print(
            "No documents to upload."
        )

        return

    print(
        f"Uploading {len(documents)} "
        f"retrieval units..."
    )

    vector_store.add_documents(
        documents
    )

    print()

    print(
        "Upload completed successfully."
    )


# ============================================================
# SUMMARY
# ============================================================

def print_summary(
    documents,
):

    upanishad_stats = {}

    for document in documents:

        metadata = document.metadata

        name = metadata.get(
            "upanishad",
            "Unknown",
        )

        if name not in upanishad_stats:

            upanishad_stats[name] = 0

        upanishad_stats[name] += 1

    print()
    print("=" * 70)
    print("INGESTION SUMMARY")
    print("=" * 70)

    print(
        f"Collection : {COLLECTION_NAME}"
    )

    print(
        f"Total units: {len(documents)}"
    )

    print()

    print(
        "Retrieval units by Upanishad:"
    )

    for name, count in upanishad_stats.items():

        print(
            f"  {name}: {count}"
        )

    print()

    print(
        "Upanishad authors:"
    )

    for filename, info in UPANISHAD_INFO.items():

        print(
            f"  {info['name']} "
            f"→ {info['author']}"
        )

    print()

    print(
        "Ingestion finished successfully."
    )

    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main():

    # 1. Load and structure source material
    documents = load_documents()

    # 2. Load embeddings
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
    print_summary(
        documents
    )


if __name__ == "__main__":
    main()