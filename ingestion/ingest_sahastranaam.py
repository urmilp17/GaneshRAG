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

DOCS_DIR = BASE_DIR / "docs" / "sahastranaam"

COLLECTION_NAME = "sahastranaam"


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

SOURCE_NAME = "Ganesh Sahastranaam"

COMMENTARY_NAME = "Khadyota"

COMMENTATOR = "Bhaskararaya Dikshita"

SOURCE_TYPE = "sahastranaam_commentary"

DOCUMENT_TYPE = "commentary"

AUTHORITY = "traditional_commentary"

TRADITION = "Ganapatya"


# ============================================================
# FILE ORDER
# ============================================================

# Explicit ordering is used because filesystem ordering is not
# guaranteed to follow Intro → First → Second → ... → Tenth → Outro.

FILE_ORDER = {
    "Ganesh_Sahastranaam_Intro.txt": 0,
    "Ganesh_Sahastranaam_First.txt": 1,
    "Ganesh_Sahastranaam_Second.txt": 2,
    "Ganesh_Sahastranaam_Third.txt": 3,
    "Ganesh_Sahastranaam_Fourth.txt": 4,
    "Ganesh_Sahastranaam_Fifth.txt": 5,
    "Ganesh_Sahastranaam_Sixth.txt": 6,
    "Ganesh_Sahastranaam_Seventh.txt": 7,
    "Ganesh_Sahastranaam_Eighth.txt": 8,
    "Ganesh_Sahastranaam_Ninth.txt": 9,
    "Ganesh_Sahastranaam_Tenth.txt": 10,
    "Ganesh_Sahastranaam_Outro.txt": 11,
}


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def clean_text(text: str) -> str:
    """
    Perform conservative text normalization.

    Sanskrit, transliteration and English commentary are preserved.
    This function only removes formatting noise.
    """

    # Normalize line endings
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Remove trailing whitespace
    text = "\n".join(
        line.rstrip()
        for line in text.splitlines()
    )

    # Prevent excessive blank lines
    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    return text.strip()


# ============================================================
# FILE TITLE
# ============================================================

def get_file_title(file_path: Path) -> str:
    """
    Convert filename into a readable title.

    Example:

    Ganesh_Sahastranaam_First.txt

    ->
    
    Ganesh Sahastranaam First
    """

    title = file_path.stem

    title = title.replace("_", " ")

    title = re.sub(
        r"\s+",
        " ",
        title
    )

    return title.strip()


# ============================================================
# FILE SECTION
# ============================================================

def get_file_section(file_path: Path) -> str:
    """
    Determine the logical section represented by the file.
    """

    name = file_path.stem.lower()

    if "intro" in name:
        return "Introduction"

    if "outro" in name:
        return "Conclusion"

    match = re.search(
        r"ganesh_sahastranaam_(first|second|third|fourth|fifth|"
        r"sixth|seventh|eighth|ninth|tenth)",
        name
    )

    if match:
        return match.group(1).capitalize()

    return file_path.stem


# ============================================================
# SAHASTRANAAM NAME EXTRACTION
# ============================================================

def extract_name_from_heading(line: str):
    """
    Extract a Sahastranaam name from a heading such as:

        Shloka 1: Gaṇeśvaraḥ (गणेश्वरः)

    Returns:

        Gaṇeśvaraḥ

    The Sanskrit name in parentheses is preserved separately
    through the metadata when available.
    """

    pattern = re.compile(
        r"Shloka\s+\d+\s*:\s*"
        r"(.+?)"
        r"\s*\(([^)]+)\)",
        re.IGNORECASE
    )

    match = pattern.search(line.strip())

    if not match:
        return None, None

    transliterated_name = match.group(1).strip()
    devanagari_name = match.group(2).strip()

    return transliterated_name, devanagari_name


# ============================================================
# CHUNKING
# ============================================================

def split_into_name_sections(text: str):
    """
    Split the commentary primarily around individual
    Sahastranaam name headings.

    This is preferable to blindly splitting every N characters,
    because each name and its commentary form a meaningful
    retrieval unit.

    Example:

        Shloka 1: Gaṇeśvaraḥ (गणेश्वरः)
        Sanskrit verse
        English commentary

        Shloka 2: ...

    becomes separate retrieval documents.
    """

    lines = text.splitlines()

    sections = []

    current_lines = []
    current_name = None
    current_devanagari = None
    current_shloka_number = None

    heading_pattern = re.compile(
        r"Shloka\s+(\d+)\s*:\s*",
        re.IGNORECASE
    )

    for line in lines:

        heading_match = heading_pattern.match(
            line.strip()
        )

        if heading_match:

            # Save previous section
            if current_lines:

                sections.append(
                    {
                        "shloka_number": current_shloka_number,
                        "name": current_name,
                        "name_devanagari": current_devanagari,
                        "text": "\n".join(
                            current_lines
                        ).strip(),
                    }
                )

            # Start new section
            current_shloka_number = int(
                heading_match.group(1)
            )

            name, devanagari = extract_name_from_heading(
                line
            )

            current_name = name
            current_devanagari = devanagari

            current_lines = [line]

        else:
            current_lines.append(line)

    # Save final section
    if current_lines:

        sections.append(
            {
                "shloka_number": current_shloka_number,
                "name": current_name,
                "name_devanagari": current_devanagari,
                "text": "\n".join(
                    current_lines
                ).strip(),
            }
        )

    return sections


# ============================================================
# CHUNK ID
# ============================================================

def create_chunk_id(
    file_name: str,
    shloka_number,
    name: str,
    content: str,
) -> str:

    raw = (
        f"{SOURCE_NAME}|"
        f"{COMMENTARY_NAME}|"
        f"{file_name}|"
        f"{shloka_number}|"
        f"{name}|"
        f"{content}"
    )

    return hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()


# ============================================================
# CITATION
# ============================================================

def create_citation(
    section: str,
    shloka_number,
    name: str,
) -> str:

    if shloka_number is not None and name:

        return (
            f"({SOURCE_NAME}, "
            f"{COMMENTARY_NAME}, "
            f"{name}, "
            f"Shloka {shloka_number})"
        )

    return (
        f"({SOURCE_NAME}, "
        f"{COMMENTARY_NAME}, "
        f"{section})"
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

    # Sort using explicit file order
    files.sort(
        key=lambda path: FILE_ORDER.get(
            path.name,
            999
        )
    )

    print("=" * 70)
    print("GANESH SAHASTRANAAM — KHADYOTA INGESTION")
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

    print(
        f"Commentary       : {COMMENTARY_NAME}"
    )

    print(
        f"Commentator      : {COMMENTATOR}"
    )

    print()

    documents = []

    for file_path in files:

        print(
            f"Reading: {file_path.name}"
        )

        try:

            text = file_path.read_text(
                encoding="utf-8"
            )

        except UnicodeDecodeError:

            text = file_path.read_text(
                encoding="utf-8-sig"
            )

        text = clean_text(text)

        if not text:

            print(
                "  WARNING: Empty file — skipped."
            )

            continue

        file_title = get_file_title(
            file_path
        )

        section = get_file_section(
            file_path
        )

        # ----------------------------------------------------
        # Split by individual Sahastranaam name
        # ----------------------------------------------------

        name_sections = split_into_name_sections(
            text
        )

        # ----------------------------------------------------
        # Intro / Outro may not contain Shloka headings
        # ----------------------------------------------------

        if not name_sections:

            name_sections = [
                {
                    "shloka_number": None,
                    "name": None,
                    "name_devanagari": None,
                    "text": text,
                }
            ]

        print(
            f"  Section : {section}"
        )

        print(
            f"  Retrieval units: {len(name_sections)}"
        )

        # ----------------------------------------------------
        # Create Documents
        # ----------------------------------------------------

        for chunk_index, section_data in enumerate(
            name_sections
        ):

            content = section_data["text"]

            shloka_number = section_data[
                "shloka_number"
            ]

            name = section_data["name"]

            name_devanagari = section_data[
                "name_devanagari"
            ]

            chunk_id = create_chunk_id(
                file_name=file_path.name,
                shloka_number=shloka_number,
                name=name,
                content=content,
            )

            citation = create_citation(
                section=section,
                shloka_number=shloka_number,
                name=name,
            )

            metadata = {
                # ------------------------------------------------
                # Collection
                # ------------------------------------------------
                "collection": COLLECTION_NAME,

                # ------------------------------------------------
                # Source classification
                # ------------------------------------------------
                "source_type": SOURCE_TYPE,
                "document_type": DOCUMENT_TYPE,

                # ------------------------------------------------
                # Source
                # ------------------------------------------------
                "source": SOURCE_NAME,
                "source_name": SOURCE_NAME,
                "text_source": SOURCE_NAME,

                # ------------------------------------------------
                # Commentary
                # ------------------------------------------------
                "commentary": COMMENTARY_NAME,
                "commentator": COMMENTATOR,

                # ------------------------------------------------
                # Tradition / authority
                # ------------------------------------------------
                "authority": AUTHORITY,
                "tradition": TRADITION,

                # ------------------------------------------------
                # File
                # ------------------------------------------------
                "document_title": file_title,
                "file_name": file_path.name,
                "file_path": str(
                    file_path.relative_to(BASE_DIR)
                ),

                # ------------------------------------------------
                # Sahastranaam structure
                # ------------------------------------------------
                "section": section,

                "name": name or "",
                "name_devanagari": (
                    name_devanagari or ""
                ),

                "shloka_number": (
                    shloka_number
                    if shloka_number is not None
                    else -1
                ),

                # ------------------------------------------------
                # Chunk information
                # ------------------------------------------------
                "chunk_id": chunk_id,
                "chunk_index": chunk_index,
                "total_chunks": len(
                    name_sections
                ),

                # ------------------------------------------------
                # Citation
                # ------------------------------------------------
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
# CREATE EMBEDDING MODEL
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
# ASTRA DB
# ============================================================

def create_vector_store(
    embeddings
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
# UPLOAD
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
        f"Uploading {len(documents)} retrieval units..."
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
    documents
):

    file_names = set()

    name_count = 0
    intro_count = 0
    outro_count = 0

    for document in documents:

        metadata = document.metadata

        file_names.add(
            metadata.get(
                "file_name",
                ""
            )
        )

        if metadata.get("name"):

            name_count += 1

        section = metadata.get(
            "section",
            ""
        )

        if section == "Introduction":

            intro_count += 1

        elif section == "Conclusion":

            outro_count += 1

    print()
    print("=" * 70)
    print("INGESTION SUMMARY")
    print("=" * 70)

    print(
        f"Collection       : {COLLECTION_NAME}"
    )

    print(
        f"Source            : {SOURCE_NAME}"
    )

    print(
        f"Commentary        : {COMMENTARY_NAME}"
    )

    print(
        f"Commentator       : {COMMENTATOR}"
    )

    print(
        f"Files processed   : {len(file_names)}"
    )

    print(
        f"Name/commentary units : {name_count}"
    )

    print(
        f"Introduction units : {intro_count}"
    )

    print(
        f"Conclusion units   : {outro_count}"
    )

    print(
        f"Total uploaded     : {len(documents)}"
    )

    print()

    print(
        "Collection is ready for retrieval."
    )

    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main():

    # 1. Read source files and create
    #    name-level retrieval units
    documents = load_documents()

    # 2. Load embedding model
    embeddings = create_embeddings()

    # 3. Connect to Astra DB
    vector_store = create_vector_store(
        embeddings
    )

    # 4. Upload documents
    ingest_documents(
        vector_store,
        documents,
    )

    # 5. Print summary
    print_summary(
        documents
    )


if __name__ == "__main__":
    main()