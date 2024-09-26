import os
import PyPDF2
import json

# Function to read PDFs and extract text
def read_pdf(pdf_file_path):
    text = ""
    with open(pdf_file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text.strip()

# Function to extract department and university names dynamically
def extract_department_university(text):
    # Assuming the university and department names are at the beginning of the PDF
    lines = text.split("\n")
    
    university_name = None
    department_name = None
    
    for line in lines:
        if "University" in line:
            university_name = line.strip()
        if "Department" in line:
            department_name = line.strip()
        
        # Stop after finding both university and department names
        if university_name and department_name:
            break
    
    return university_name, department_name

# Function to parse text into sections based on keywords
def parse_pdf_text_to_structure(pdf_text):
    sections = {
        "introduction": "",
        "vision": "",
        "mission": "",
        "goals": {
            "undergraduate": [],
            "graduate": []
        },
        "academic_programs": {
            "currently_offering": [],
            "proposed": []
        }
    }
    
    # Split the text into lines
    lines = pdf_text.split("\n")
    
    # Track which section we're in
    current_section = None
    
    # Loop through lines and categorize them based on keywords
    for line in lines:
        line = line.strip()
        
        if "Introduction" in line:
            current_section = "introduction"
        elif "Vision" in line:
            current_section = "vision"
        elif "Mission" in line:
            current_section = "mission"
        elif "Undergraduate Program" in line:
            current_section = "undergraduate_goals"
        elif "Graduate Program" in line:
            current_section = "graduate_goals"
        elif "Currently Offering" in line:
            current_section = "currently_offering"
        elif "Proposed" in line:
            current_section = "proposed"
        else:
            # Add the line to the correct section
            if current_section == "introduction":
                sections["introduction"] += line + " "
            elif current_section == "vision":
                sections["vision"] += line + " "
            elif current_section == "mission":
                sections["mission"] += line + " "
            elif current_section == "undergraduate_goals":
                sections["goals"]["undergraduate"].append(line)
            elif current_section == "graduate_goals":
                sections["goals"]["graduate"].append(line)
            elif current_section == "currently_offering":
                sections["academic_programs"]["currently_offering"].append(line)
            elif current_section == "proposed":
                sections["academic_programs"]["proposed"].append(line)

    return sections

# Save structured data to a JSON file
def save_to_json(file_path, data):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

# Main function to process all PDFs in a directory
def process_pdfs_in_directory(pdf_directory):
    all_structured_data = []
    
    for filename in os.listdir(pdf_directory):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(pdf_directory, filename)
            print(f"Processing file: {filename}")

            # Read the PDF
            pdf_text = read_pdf(pdf_path)
            
            # Dynamically extract university and department names
            university_name, department_name = extract_department_university(pdf_text)
            
            # Parse the PDF text into structured data
            structured_data = {
                "department": {
                    "name": department_name if department_name else "Unknown Department",
                    "university": university_name if university_name else "Unknown University",
                    **parse_pdf_text_to_structure(pdf_text)  # Merge parsed sections into department structure
                }
            }

            # Append structured data to the list
            all_structured_data.append(structured_data)

            # Save individual JSON for each PDF
            output_json_file_path = os.path.join(pdf_directory, f"{os.path.splitext(filename)[0]}.json")
            save_to_json(output_json_file_path, structured_data)
            print(f"Data saved to {output_json_file_path}")

    # Save all structured data into a single JSON file
    final_output_json_file_path = os.path.join(pdf_directory, 'all_structured_data.json')
    save_to_json(final_output_json_file_path, all_structured_data)
    print(f"All data saved to {final_output_json_file_path}")

# Example usage
pdf_directory_path = 'D:\Project\FYP\pdf-bot\data'  # Update with your PDF folder path
process_pdfs_in_directory(pdf_directory_path)
