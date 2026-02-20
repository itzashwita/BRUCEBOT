from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
import requests, os, urllib.parse

BASE_URL = "https://crackap.com/questions.php?type=csp"
OUTPUT_DIR = "csp_pdfs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def download_file(url):
    name = url.split("/")[-1]
    path = os.path.join(OUTPUT_DIR, name)
    r = requests.get(url)
    with open(path, "wb") as f:
        f.write(r.content)
    print("Downloaded:", name)

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    page.goto(BASE_URL, wait_until="networkidle")
    html = page.content()
    browser.close()

soup = BeautifulSoup(html, "html.parser")

for link in soup.find_all("a", href=True):
    href = urllib.parse.urljoin(BASE_URL, link["href"])
    if href.endswith(".pdf"):
        download_file(href)
