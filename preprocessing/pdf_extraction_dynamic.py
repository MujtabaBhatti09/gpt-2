# import os
# import PyPDF2
# import re

# def extract_text_from_pdf(pdf_file):
#     """Extracts text from a PDF file."""
#     with open(pdf_file, 'rb') as file:
#         reader = PyPDF2.PdfReader(file)
#         text = ''
#         for page in reader.pages:
#             text += page.extract_text()
#         return text

# def filter_relevant_content(text):
#     """Filter out sections like 'Vision', 'Objectives' that are related to GCUH."""
#     relevant_sections = ['Vision', 'Objectives', 'Introduction', 'Faculty Members']
#     filtered_content = ""
    
#     for section in relevant_sections:
#         pattern = rf"{section}.*"
#         match = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
#         if match:
#             filtered_content += "\n".join(match)
    
#     return filtered_content

# # Use this function to filter out important sections from each PDF and re-train the model
# pdf_files = ['../data/GCUH-Mathematics.pdf', '../data/GCUH-ORIC.pdf']  # Add all your PDFs here

# for pdf_file in pdf_files:
#     text = extract_text_from_pdf(pdf_file)
#     filtered_text = filter_relevant_content(text)
#     with open("../data/filtered_dataset.txt", "a", encoding="utf-8") as f:
#         f.write(filtered_text)

# import os
# import re
# import PyPDF2

# def extract_text_from_pdf(pdf_file):
#     """Extracts text from a PDF file."""
#     with open(pdf_file, 'rb') as file:
#         reader = PyPDF2.PdfReader(file)
#         text = ''
#         for page in reader.pages:
#             text += page.extract_text()
#         return text

# def dynamic_section_detection(text):
#     """Dynamically detects sections and text content using simple heuristics like capitalized words, headings, and paragraphs."""
#     sections = {}
#     current_section = None

#     # Example heuristic to detect section headers
#     section_pattern = re.compile(r'^[A-Z][A-Za-z\s]+[:]*$')  # Detect capitalized headings

#     for line in text.split('\n'):
#         line = line.strip()

#         # Check if the line matches the section header pattern
#         if section_pattern.match(line):
#             current_section = line
#             sections[current_section] = ''
#         elif current_section:
#             # Append content to the current section
#             sections[current_section] += line + ' '

#     return sections

# def save_filtered_data(sections, output_file):
#     """Saves the extracted sections and content to a file."""
#     with open(output_file, 'w', encoding='utf-8') as f:
#         for section, content in sections.items():
#             f.write(f"SECTION: {section}\n")
#             f.write(f"{content}\n\n")

# def process_pdfs(directory):
#     """Processes all PDF files in the given directory."""
#     output_file = '../data/filtered_dataset.txt'  # Output file for the filtered dataset

#     for filename in os.listdir(directory):
#         if filename.endswith(".pdf"):
#             file_path = os.path.join(directory, filename)
#             print(f"Processing {file_path}...")

#             text = extract_text_from_pdf(file_path)
#             sections = dynamic_section_detection(text)
#             save_filtered_data(sections, output_file)

# if __name__ == "__main__":
#     process_pdfs('../data')  # Adjust the directory to your PDFs folder

import os
import re
import PyPDF2

# Set the absolute path to the data folder containing multiple PDFs
PDF_DIRECTORY = r'D:\Project\FYP\pdf-bot\data'
OUTPUT_FILE = r'D:\Project\FYP\pdf-bot\data\filtered_dataset.txt'

def extract_text_from_pdf(pdf_file):
    """Extracts text from a PDF file."""
    with open(pdf_file, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text()
        return text

def dynamic_section_detection(text):
    """Dynamically detects sections and text content using simple heuristics like capitalized words, headings, and paragraphs."""
    sections = {}
    current_section = None

    # Example heuristic to detect section headers
    section_pattern = re.compile(r'^[A-Z][A-Za-z\s]+[:]*$')  # Detect capitalized headings

    for line in text.split('\n'):
        line = line.strip()

        # Check if the line matches the section header pattern
        if section_pattern.match(line):
            current_section = line
            sections[current_section] = ''
        elif current_section:
            # Append content to the current section
            sections[current_section] += line + ' '

    return sections

def save_filtered_data(sections, output_file):
    """Saves the extracted sections and content to a file."""
    with open(output_file, 'a', encoding='utf-8') as f:  # 'a' for appending content from multiple PDFs
        for section, content in sections.items():
            f.write(f"SECTION: {section}\n")
            f.write(f"{content}\n\n")

def process_pdfs(directory):
    """Processes all PDF files in the given directory."""
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            file_path = os.path.join(directory, filename)
            print(f"Processing {file_path}...")

            text = extract_text_from_pdf(file_path)
            sections = dynamic_section_detection(text)
            save_filtered_data(sections, OUTPUT_FILE)

if __name__ == "__main__":
    # Clear the file before appending new content
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write("")  # Clears the existing file content

    process_pdfs(PDF_DIRECTORY)  # Access the PDF directory using absolute path