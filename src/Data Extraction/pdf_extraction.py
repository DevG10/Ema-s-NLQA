import os
import requests
import fitz


urls = [
    "https://introml.mit.edu/_static/spring24/LectureNotes/chapter_Transformers.pdf",
    "https://web.stanford.edu/class/cs124/lec/LLM2024.pdf",
    "https://princeton-nlp.github.io/cos484/lectures/lec18.pdf",
    "https://web.stanford.edu/class/cs329t/slides/lecture_6.pdf"
    ]


with open("Data/pdf_extracted_texts.txt", "wb") as out:
    for url in urls:
        response = requests.get(url)
        response.raise_for_status()
        doc = fitz.open(stream= response.content, filetype = 'pdf')
        for page in doc:
            text = page.get_text()
            out.write(text.encode('utf-8'))
        doc.close()
