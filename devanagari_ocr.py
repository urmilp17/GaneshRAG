#!/usr/bin/env python3
"""
Devanagari OCR Script for PDF Files
Takes a PDF as input and extracts Devanagari text to a .txt file
"""

import os
import sys
from PIL import Image
import pytesseract
import cv2
import numpy as np
from pdf2image import convert_from_path
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from rich.panel import Panel
from rich.text import Text
from rich.traceback import install
import time

pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
# Install rich traceback for better error messages
install(show_locals=True, width=120, extra_lines=3, word_wrap=True)

# Initialize Rich Console
console = Console()

def preprocess_image(image):
    """
    Preprocess the image to improve OCR accuracy for Devanagari text
    """
    # Convert PIL Image to OpenCV format
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to get black and white image
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Denoise
    denoised = cv2.medianBlur(thresh, 3)
    
    # Dilate to connect broken characters
    kernel = np.ones((1, 1), np.uint8)
    dilated = cv2.dilate(denoised, kernel, iterations=1)
    
    # Convert back to PIL Image
    return Image.fromarray(dilated)

def perform_ocr_on_image(image, lang=None):
    """
    Perform OCR on a single image and return extracted text
    """
    try:
        # Preprocess the image
        processed_img = preprocess_image(image)
        
        # Perform OCR
        text = pytesseract.image_to_string(
            processed_img, 
            lang=lang,
            config='--psm 6 --oem 3'  # PSM 6: Uniform block of text, OEM 3: Default LSTM
        )
        
        return text.strip()
    
    except Exception as e:
        console.print(f"[red]Error during OCR: {e}[/red]")
        return ""

def process_pdf_pages(pdf_path, lang=None, dpi=300):
    """
    Convert PDF pages to images and process each page with OCR
    Yields page number and extracted text
    """
    try:
        # Convert PDF to images
        console.print(f"[cyan]Converting PDF to images...[/cyan]")
        images = convert_from_path(pdf_path, dpi=dpi)
        
        total_pages = len(images)
        console.print(f"[green]✓ PDF loaded successfully: {total_pages} pages found[/green]")
        
        # Process each page
        for page_num, image in enumerate(images, start=1):
            console.print(f"\n[yellow]📄 Processing page {page_num}/{total_pages}...[/yellow]")
            
            # Perform OCR on the image
            text = perform_ocr_on_image(image, lang)
            
            # Show preview of extracted text
            if text:
                preview = text[:200] + "..." if len(text) > 200 else text
                console.print(f"[dim]Preview: {preview.replace(chr(10), ' ')}[/dim]")
                console.print(f"[green]✓ Extracted {len(text)} characters from page {page_num}[/green]")
            else:
                console.print(f"[yellow]⚠ No text found on page {page_num}[/yellow]")
            
            yield page_num, text, total_pages
            
    except Exception as e:
        console.print(f"[red]Error processing PDF: {e}[/red]")
        raise

def save_text_to_file(text, output_file):
    """
    Save extracted text to a .txt file
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(text)
        console.print(f"\n[green]✓ Text successfully saved to: {output_file}[/green]")
        return True
    except Exception as e:
        console.print(f"[red]Error saving file: {e}[/red]")
        return False

def get_user_input():
    """
    Get input and output file names from user with rich console prompts
    """
    console.print(Panel.fit(
        "[bold cyan]📝 Devanagari OCR PDF Processor[/bold cyan]\n"
        "[dim]Extract Devanagari text from PDF files[/dim]",
        border_style="cyan"
    ))
    
    # Get input PDF file
    while True:
        console.print("\n[bold]Enter the PDF file path:[/bold]")
        pdf_path = input().strip().strip('"').strip("'")
        
        if not pdf_path:
            console.print("[red]❌ File path cannot be empty. Please try again.[/red]")
            continue
            
        if not os.path.exists(pdf_path):
            console.print(f"[red]❌ File '{pdf_path}' not found. Please check the path and try again.[/red]")
            continue
            
        # Check if it's a PDF file
        if not pdf_path.lower().endswith('.pdf'):
            console.print("[yellow]⚠ Warning: File does not have .pdf extension. Continuing anyway...[/yellow]")
            
        break
    
    # Get output text file
    while True:
        console.print("\n[bold]Enter the output text file name (e.g., output.txt):[/bold]")
        output_file = input().strip().strip('"').strip("'")
        
        if not output_file:
            console.print("[red]❌ Output file name cannot be empty. Please try again.[/red]")
            continue
            
        # Add .txt extension if not present
        if not output_file.lower().endswith('.txt'):
            output_file += '.txt'
            
        # Check if file exists and ask for overwrite
        if os.path.exists(output_file):
            console.print(f"[yellow]⚠ File '{output_file}' already exists.[/yellow]")
            overwrite = input("Do you want to overwrite it? (y/n): ").strip().lower()
            if overwrite != 'y':
                console.print("[cyan]Please enter a different file name.[/cyan]")
                continue
        break
    
    # Get language choice
    console.print("\n[bold]Select OCR language:[/bold]")
    languages = {
        '1': ('hin+eng', 'Hindi'),
        '2': ('san+eng', 'Sanskrit'),
        '3': ('mar', 'Marathi'),
        '4': ('nep', 'Nepali'),
        '5': ('eng','English')
    }
    
    for key, (code, name) in languages.items():
        console.print(f"  {key}. {name} ({code})")
    
    while True:
        choice = input("\nEnter your choice (1-4) [default: 1]: ").strip()
        if not choice:
            choice = '1'
        if choice in languages:
            lang_code, lang_name = languages[choice]
            console.print(f"[green]Selected: {lang_name} ({lang_code})[/green]")
            break
        else:
            console.print("[red]Invalid choice. Please enter 1-4.[/red]")
    
    # Get DPI setting
    console.print("\n[bold]Enter DPI for image conversion (higher = better quality but slower)[/bold]")
    console.print("[dim]Recommended: 300 for good quality, 150 for faster processing[/dim]")
    dpi_input = input("DPI [default: 300]: ").strip()
    dpi = int(dpi_input) if dpi_input.isdigit() else 300
    console.print(f"[green]Using DPI: {dpi}[/green]")
    
    return pdf_path, output_file, lang_code, dpi

def display_summary(pdf_path, output_file, total_pages, total_chars, processing_time):
    """
    Display processing summary in a rich table
    """
    table = Table(title="Processing Summary", title_style="bold cyan")
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="green")
    
    table.add_row("Input File", os.path.basename(pdf_path))
    table.add_row("Output File", output_file)
    table.add_row("Total Pages Processed", str(total_pages))
    table.add_row("Total Characters Extracted", f"{total_chars:,}")
    table.add_row("Processing Time", f"{processing_time:.2f} seconds")
    
    console.print(table)

def main():
    try:
        # Get user input
        pdf_path, output_file, lang_code, dpi = get_user_input()
        
        # Start processing
        console.print("\n[bold cyan]🚀 Starting PDF OCR Processing...[/bold cyan]")
        start_time = time.time()
        
        # Collect all text
        all_text = []
        total_chars = 0
        total_pages = 0
        
        # Process PDF pages
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console,
            transient=False,
        ) as progress:
            
            task = progress.add_task("[cyan]Processing PDF...", total=None)
            
            for page_num, text, total in process_pdf_pages(pdf_path, lang_code, dpi):
                total_pages = total
                progress.update(task, total=total, completed=page_num)
                
                # Add page separator and text
                if text:
                    all_text.append(f"--- Page {page_num} ---\n{text}\n")
                    total_chars += len(text)
                else:
                    all_text.append(f"--- Page {page_num} ---\n[No text found]\n")
        
        # Combine all text
        final_text = "\n".join(all_text)
        
        # Save to file
        if save_text_to_file(final_text, output_file):
            processing_time = time.time() - start_time
            display_summary(pdf_path, output_file, total_pages, total_chars, processing_time)
            
            # Show sample of extracted text
            if final_text.strip():
                console.print("\n[bold cyan]📝 Sample of extracted text:[/bold cyan]")
                sample = final_text[:500] + "..." if len(final_text) > 500 else final_text
                console.print(Panel(sample, border_style="green"))
            else:
                console.print("\n[yellow]⚠ No text was extracted from any page.[/yellow]")
        
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠ Process interrupted by user.[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[red]❌ An unexpected error occurred: {e}[/red]")
        import traceback
        console.print(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()