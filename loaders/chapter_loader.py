import os
import re
from typing import List, Optional
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document

class ChapterLoader:
    """
    A class to load and split documents into chapters from text files.
    """
    
    def __init__(self, chapter_pattern: str = r'(Chapter\s+\d+\s*[:.]?\s*[^\n]+)'):
        """
        Initialize the ChapterLoader with a pattern for chapter headings.
        
        Args:
            chapter_pattern: Regex pattern to identify chapter headings
        """
        self.chapter_pattern = chapter_pattern
    
    def load_and_split_chapters(self, folder_path: str, book_name: str) -> List[Document]:
        """
        Load all text files from a folder and split them into chapters.
        
        Args:
            folder_path: Path to the folder containing text files
            book_name: Name of the book (e.g., "Ganesh_Puran_Upasana_Khand")
        
        Returns:
            List of Document objects, each representing a chapter
        """
        documents = []
        
        # Get all text files in the folder
        text_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
        
        for file_name in text_files:
            file_path = os.path.join(folder_path, file_name)
            docs = self.load_and_split_chapters_for_file(file_path, book_name)
            documents.extend(docs)
        
        return documents
    
    def load_and_split_chapters_for_file(self, file_path: str, book_name: str) -> List[Document]:
        """
        Load and split a single file into chapters.
        
        Args:
            file_path: Path to the text file
            book_name: Name of the book
        
        Returns:
            List of Document objects, each representing a chapter
        """
        documents = []
        
        # Load the document
        loader = TextLoader(file_path, encoding='utf-8')
        loaded_docs = loader.load()
        full_text = loaded_docs[0].page_content
        
        # Split by chapter headings
        chapter_matches = list(re.finditer(self.chapter_pattern, full_text, re.IGNORECASE))
        
        if not chapter_matches:
            # If no chapters found, treat the whole file as one document
            doc = Document(
                page_content=full_text,
                metadata={
                    "book": book_name,
                    # "file": os.path.basename(file_path),
                    "chapter_info": "Full text"
                }
            )
            documents.append(doc)
        else:
            # Split the text by chapters
            for i, match in enumerate(chapter_matches):
                chapter_heading = match.group(1).strip()
                start_pos = match.start()
                
                if i == len(chapter_matches) - 1:
                    end_pos = len(full_text)
                else:
                    end_pos = chapter_matches[i + 1].start()
                
                chapter_content = full_text[start_pos:end_pos].strip()
                
                doc = Document(
                    page_content=chapter_content,
                    metadata={
                        "book": book_name,
                        # "file": os.path.basename(file_path),
                        "chapter_info": chapter_heading
                    }
                )
                documents.append(doc)
        
        return documents
    
    def load_folder_books(self, folder_path: str, book_names: Optional[List[str]] = None) -> List[Document]:
        """
        Load multiple books from a folder.
        
        Args:
            folder_path: Path to the folder containing text files
            book_names: List of book names (without .txt extension). If None, loads all .txt files
        
        Returns:
            List of Document objects
        """
        all_documents = []
        
        if book_names is None:
            # Load all text files in the folder
            text_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
            book_names = [f.replace('.txt', '') for f in text_files]
        
        for book_name in book_names:
            file_path = os.path.join(folder_path, f"{book_name}.txt")
            if os.path.exists(file_path):
                docs = self.load_and_split_chapters_for_file(file_path, book_name)
                all_documents.extend(docs)
                print(f"Loaded {len(docs)} chapters from {book_name}")
        
        return all_documents
    
    def load_from_paths(self, paths: dict) -> List[Document]:
        """
        Load documents from multiple folder paths.
        
        Args:
            paths: Dictionary mapping folder_path to list of book_names
                  e.g., {"docs/ganesh_puran": ["Ganesh_Puran_Krida_Khand", "Ganesh_Puran_Upasana_Khand"]}
        
        Returns:
            List of Document objects
        """
        all_documents = []
        
        for folder_path, book_names in paths.items():
            print(f"\nLoading from folder: {folder_path}")
            docs = self.load_folder_books(folder_path, book_names)
            all_documents.extend(docs)
            print(f"Total chapters from this folder: {len(docs)}")
        
        return all_documents