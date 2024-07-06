import PyPDF2
from transformers import pipeline

def read_pdf(file_path):
    pdf_reader = PyPDF2.PdfFileReader(open(file_path, 'rb'))
    text = ""
    for page_num in range(pdf_reader.getNumPages()):
        page = pdf_reader.getPage(page_num)
        text += page.extract_text()
    return text

def summarize_pdf(file_path):
    text = read_pdf(file_path)
    summarizer = pipeline('summarization')
    summary = summarizer(text, max_length=150, min_length=30, do_sample=False)
    return summary[0]['summary_text']
