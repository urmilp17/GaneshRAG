import os
import re
import hashlib
from pathlib import Path

from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_astradb import AstraDBVectorStore

from embedder import SentenceTransformerEmbeddings


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv(override=True)


# ------------------------------------------------------------
# Project paths
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DOCS_DIR = (
    BASE_DIR
    / "docs"
    / "iconography"
)


# ------------------------------------------------------------
# Astra DB
# ------------------------------------------------------------

COLLECTION_NAME = "iconography"

ASTRA_TOKEN = os.getenv(
    "ASTRA_DB_APPLICATION_TOKEN"
)

ASTRA_ENDPOINT = os.getenv(
    "ASTRA_DB_API_ENDPOINT"
)


# ------------------------------------------------------------
# Embedding model
# ------------------------------------------------------------

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# ============================================================
# VALIDATION
# ============================================================

if not ASTRA_TOKEN:

    raise ValueError(
        "ASTRA_DB_APPLICATION_TOKEN "
        "is not configured in .env"
    )


if not ASTRA_ENDPOINT:

    raise ValueError(
        "ASTRA_DB_API_ENDPOINT "
        "is not configured in .env"
    )


if not DOCS_DIR.exists():

    raise FileNotFoundError(
        f"Iconography directory not found: "
        f"{DOCS_DIR}"
    )


# ============================================================
# TEXT CLEANING
# ============================================================

def remove_numeric_values(text: str) -> str:
    """
    Remove numerical values from the text.

    Examples:

        '12. Ekadanta'
            -> 'Ekadanta'

        'Chapter 4'
            -> 'Chapter'

        'Page 27'
            -> 'Page'

        '124 forms'
            -> 'forms'

    This is intentionally conservative regarding
    punctuation and Sanskrit terminology.
    """

    # --------------------------------------------------------
    # Remove numbers with optional decimal values
    #
    # Examples:
    #   12
    #   12.
    #   12.5
    #   12.5.
    # --------------------------------------------------------

    text = re.sub(
        r"\b\d+(?:\.\d+)?\.?\b",
        "",
        text
    )

    # --------------------------------------------------------
    # Remove extra whitespace
    # --------------------------------------------------------

    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    # --------------------------------------------------------
    # Remove spaces before punctuation
    # --------------------------------------------------------

    text = re.sub(
        r"\s+([,.;:!?])",
        r"\1",
        text
    )

    # --------------------------------------------------------
    # Clean excessive blank lines
    # --------------------------------------------------------

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    return text.strip()


# ============================================================
# FORM NAME
# ============================================================

def get_form_name(file_path: Path) -> str:
    """
    Extract form name from filename.

    Example:

        Ekadanta.txt
            -> Ekadanta

        12. Ekadanta.txt
            -> Ekadanta
    """

    name = file_path.stem.strip()

    # Remove leading numbering such as:
    #
    # 12.
    # 12
    # 12.-
    # 12 - 
    # 12_Ekadanta

    name = re.sub(
        r"^\s*\d+(?:\.\d+)?[\s._-]*",
        "",
        name
    )

    return name.strip()


# ============================================================
# SOURCE NAME
# ============================================================

def get_source_name(source_folder: Path) -> str:
    """
    Return the parent source folder name.

    Examples:

        ShriTattvaNidhi
        Vinayak Rahasya
        Vinayak Tantra
    """

    return source_folder.name.strip()


# ============================================================
# DETERMINISTIC CHUNK ID
# ============================================================

def generate_chunk_id(
    source: str,
    form_name: str,
    content: str
) -> str:
    """
    Generate a deterministic ID.

    This prevents accidental duplication if the same
    ingestion script is executed again.
    """

    raw = (
        f"iconography|"
        f"{source}|"
        f"{form_name}|"
        f"{content}"
    )

    return hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()


# ============================================================
# CREATE DOCUMENT
# ============================================================

def create_document(
    file_path: Path,
    source_folder: Path
):
    """
    Convert one iconography TXT file into
    a LangChain Document.
    """

    # --------------------------------------------------------
    # Read file
    # --------------------------------------------------------

    try:

        text = file_path.read_text(
            encoding="utf-8"
        )

    except UnicodeDecodeError:

        text = file_path.read_text(
            encoding="utf-8-sig"
        )


    # --------------------------------------------------------
    # Clean text
    # --------------------------------------------------------

    cleaned_text = remove_numeric_values(
        text
    )


    # --------------------------------------------------------
    # Skip empty files
    # --------------------------------------------------------

    if not cleaned_text:

        return None


    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    source = get_source_name(
        source_folder
    )

    form_name = get_form_name(
        file_path
    )


    chunk_id = generate_chunk_id(
        source,
        form_name,
        cleaned_text
    )


    metadata = {

        # ====================================================
        # DATASET IDENTIFICATION
        # ====================================================

        "collection":
            COLLECTION_NAME,

        "source_type":
            "iconography",

        "document_type":
            "iconographic_form",


        # ====================================================
        # SOURCE
        # ====================================================

        "source":
            source,

        "source_name":
            source,

        "text_source":
            source,


        # ====================================================
        # FORM
        # ====================================================

        "form_name":
            form_name,

        "deity":
            "Ganesha",


        # ====================================================
        # FILE INFORMATION
        # ====================================================

        "file_name":
            file_path.name,

        "file_path":
            str(
                file_path.relative_to(
                    BASE_DIR
                )
            ),


        # ====================================================
        # CHUNK INFORMATION
        # ====================================================

        "chunk_id":
            chunk_id,

        "chunk_index":
            0,

        "total_chunks":
            1,


        # ====================================================
        # AUTHORITY
        # ====================================================

        "authority":
            "traditional_iconographic_source",


        # ====================================================
        # RETRIEVAL / CITATION
        # ====================================================

        "citation":
            f"{source}, {form_name}",

        "section":
            form_name,

        "chapter":
            None,

        "chapter_number":
            None,

        "page_number":
            None
    }


    return Document(

        page_content=cleaned_text,

        metadata=metadata
    )


# ============================================================
# LOAD ALL ICONOGRAPHY FILES
# ============================================================

def load_iconography_documents():

    documents = []

    source_folders = sorted(

        [
            folder

            for folder in DOCS_DIR.iterdir()

            if folder.is_dir()
        ],

        key=lambda x: x.name.lower()
    )


    print(
        f"\nFound "
        f"{len(source_folders)} "
        f"iconography sources."
    )


    for source_folder in source_folders:

        source_name = (
            source_folder.name
        )

        print(
            f"\n📚 Source: "
            f"{source_name}"
        )


        txt_files = sorted(

            source_folder.glob(
                "*.txt"
            ),

            key=lambda x: x.name.lower()
        )


        print(
            f"   Found "
            f"{len(txt_files)} "
            f"TXT files."
        )


        for file_path in txt_files:

            try:

                document = create_document(

                    file_path,

                    source_folder
                )


                if document is None:

                    print(
                        f"   ⚠️ Skipped empty: "
                        f"{file_path.name}"
                    )

                    continue


                documents.append(
                    document
                )


                print(
                    f"   ✓ "
                    f"{get_form_name(file_path)}"
                )


            except Exception as e:

                print(
                    f"   ❌ Failed: "
                    f"{file_path.name}"
                )

                print(
                    f"      {e}"
                )


    return documents


# ============================================================
# INITIALIZE EMBEDDINGS
# ============================================================

print(
    "\n🔄 Loading embedding model..."
)


embedder = SentenceTransformerEmbeddings(

    model_name=EMBEDDING_MODEL,

    device=None
)


print(
    "✅ Embedding model loaded."
)


# ============================================================
# INITIALIZE ASTRA DB
# ============================================================

print(
    "\n🔄 Connecting to Astra DB..."
)

vector_store = AstraDBVectorStore(

    collection_name=COLLECTION_NAME,

    embedding=embedder,

    token=ASTRA_TOKEN,

    api_endpoint=ASTRA_ENDPOINT
)


print(
    f"✅ Connected to Astra DB collection: "
    f"{COLLECTION_NAME}"
)


# ============================================================
# MAIN INGESTION
# ============================================================

def main():

    print(
        "\n"
        "=================================================="
    )

    print(
        "      GANESHA ICONOGRAPHY INGESTION"
    )

    print(
        "=================================================="
    )


    print(
        f"\n📂 Documents directory:"
        f"\n{DOCS_DIR}"
    )

    print(
        f"\n🗄️ Collection:"
        f"\n{COLLECTION_NAME}"
    )


    # --------------------------------------------------------
    # Load documents
    # --------------------------------------------------------

    documents = load_iconography_documents()


    if not documents:

        print(
            "\n❌ No documents found."
        )

        return


    print(
        "\n"
        "--------------------------------------------------"
    )

    print(
        f"📄 Documents ready: "
        f"{len(documents)}"
    )

    print(
        "--------------------------------------------------"
    )


    # --------------------------------------------------------
    # Display sample metadata
    # --------------------------------------------------------

    print(
        "\n🔎 Sample document metadata:"
    )

    sample = documents[0]

    for key, value in sample.metadata.items():

        print(
            f"   {key}: {value}"
        )


    # --------------------------------------------------------
    # Upload documents
    # --------------------------------------------------------

    print(
        "\n🚀 Uploading documents to Astra DB..."
    )


    vector_store.add_documents(
        documents
    )


    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------

    print(
        "\n"
        "=================================================="
    )

    print(
        "✅ ICONOGRAPHY INGESTION COMPLETE"
    )

    print(
        "=================================================="
    )

    print(
        f"\nTotal documents ingested: "
        f"{len(documents)}"
    )

    print(
        f"Collection: "
        f"{COLLECTION_NAME}"
    )

    print(
        "\nSources:"
    )


    source_counts = {}


    for document in documents:

        source = document.metadata.get(
            "source",
            "Unknown"
        )

        source_counts[source] = (
            source_counts.get(
                source,
                0
            ) + 1
        )


    for source, count in (
        source_counts.items()
    ):

        print(
            f"   • {source}: "
            f"{count} forms"
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()