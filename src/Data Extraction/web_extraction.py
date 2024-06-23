import requests
from bs4 import BeautifulSoup
import os

intro_url = 'https://stanford-cs324.github.io/winter2022/lectures/introduction/'

# There were many lectures and links but I decided to keep most important ones.
lecture_urls = [
                "https://stanford-cs324.github.io/winter2022/lectures/security/",
                "https://stanford-cs324.github.io/winter2022/lectures/training/",
                # "https://introml.mit.edu/_static/spring24/LectureNotes/chapter_Transformers.pdf",
                # "https://web.stanford.edu/class/cs124/lec/LLM2024.pdf",
                
                ]

blacklist = [
        '[document]',
        'noscript',
        'header',
        'html',
        'meta',
        'head', 
        'input',
        'script',
        'style'
    ]

# Funtion for extracting the introductory notes
def extract_intro_notes(intro_url):
    res = requests.get(intro_url)
    res.raise_for_status()
    html_page = res.content

    soup = BeautifulSoup(html_page, 'html.parser')
    text = soup.find_all(text=True)

    output = ''

    for t in text:
        if t.parent.name not in blacklist:
            output += '{} '.format(t.strip())

    with open("Data/introductory_notes.txt", "w") as file:
        file.write(output)

#similar function for extracting lecture notes
def extract_lecture_notes(lecture_urls):
    for url in lecture_urls:
        res = requests.get(url)
        res.raise_for_status()
        html_page = res.content

        soup = BeautifulSoup(html_page, 'html.parser')
        text = soup.find_all(text=True)

        output = ''
        for t in text:
            if t.parent.name not in blacklist:
                output += '{} '.format(t.strip())

        with open("Data/lecture_notes.txt", "w") as file:
            file.write(output)

def main():
    extract_intro_notes(intro_url)
    extract_lecture_notes(lecture_urls)

if __name__ == "__main__":
    main()