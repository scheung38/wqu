import os
import subprocess

def image_to_pdf(image_path, pdf_path):
    os.environ['TESSDATA_PREFIX'] = '/path/to/your/tessdata'  # replace with your tessdata path
    command = ['tesseract', image_path, pdf_path, '-l', 'eng', '–psm', '11', 'pdf']
    subprocess.run(command, check=True)

def pdf_to_text(pdf_path, text_path):
    command = ['pdftotext', '-layout', pdf_path, text_path]
    subprocess.run(command, check=True)

def main():
    image_path = 'images/toc.png'
    pdf_path = 'images/toc.pdf'
    text_path = 'output.txt'
    image_to_pdf(image_path, pdf_path)
    pdf_to_text(pdf_path, text_path)

if __name__ == '__main__':
    main()
