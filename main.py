import os
from pypdf import PdfReader
# from pyPdf import PdfFileReader as PdfReader

import gradio as gr
import warnings

# Filter out user warnings to avoid any unnecessary clutter in the output
warnings.filterwarnings("ignore", category=UserWarning)
# Define a function to save extracted text to a file
def save_text_to_file(text, file_path, page_num):
    # Get the directory path of the PDF file
    file_dir = os.path.dirname(file_path)
    # Get the name of the PDF file without the extension
    file_name_without_extension = os.path.splitext(os.path.basename(file_path))[0]
    # Construct the name of the output file with the page number
    output_file_name = f"{file_name_without_extension}_page_{page_num}.txt"
    # Construct the path to the output file
    output_file_path = os.path.join(file_dir, output_file_name)
    # Write the extracted text to the output file
    with open(output_file_path, "w", encoding='utf-8') as output_file:
        output_file.write(text)
    # Return the path to the output file
    return output_file_path

# Define a function to extract text from a PDF file
def extract_text_from_pdf(file_path, page_num, download=False):
    # Remove any extra quotes around the file path
    file_path = file_path.strip('\"')
    # Open the PDF file in read-binary mode
    with open(file_path, 'rb') as pdf_file:
        # Create a PdfReader object
        pdf_reader = PdfReader(pdf_file)
        # Get the specified page from the PDF file
        page = pdf_reader.pages[page_num]
        # Extract text from the page
        text = page.extract_text()
    # If the user has selected the download option
    if download:
        # Save the extracted text to a file
        file_path = save_text_to_file(text, file_path, page_num)
        # Return the extracted text and the path to the output file
        return text, file_path
    # If the user has not selected the download option
    else:
        # Return only the extracted text
        return text, None

# Create a Gradio interface
demo = gr.Interface(
    # Function to be called when the user submits input
    fn=extract_text_from_pdf,
    # Define input components for the interface
    inputs=[
        gr.Textbox(label='PDF file path'),
        gr.Slider(label='Page number to extract', minimum=0, maximum=100, step=1),
        gr.Checkbox(label="Download Extracted Text")
    ],
    # Define output components for the interface
    outputs=[
        gr.Textbox(label='Extracted Text'),
        gr.File(label='Download Extracted Text File')
    ],
    # Set the title and description for the interface
    title='PDF Text Extractor',
    description='Extract text from a PDF file based on the file path and page number inputs.'
)
# Launch the Gradio interface
if __name__ == '__main__':
    demo.launch()