import re

def clean_response(text: str) -> str:
    text = re.sub(r'\([\w\s\-]+\.pdf,?\s*Page\s*[\d,\s]+\)', '', text)
    text = re.sub(r'\[Source:.*?\|.*?Page:.*?\]', '', text)
    text = re.sub(r'\([^)]*\.pdf[^)]*\)', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+\n', '\n', text)
    return text.strip()