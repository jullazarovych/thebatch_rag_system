import requests
from bs4 import BeautifulSoup
import json
import time
import os
from datetime import datetime
from urllib.parse import urljoin
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter

class BatchDataCollector:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.base_url = "https://www.deeplearning.ai/the-batch/"
        self.articles = []
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
        )

    def get_article_links(self):
        try:
            response = requests.get(self.base_url)
            soup = BeautifulSoup(response.content, 'html.parser')
            issue_links = []

            for link in soup.find_all('a', href=True):
                href = link['href']
                match = re.search(r'/the-batch/issue-(\d+)/?$', href)
                if match:
                    issue_number = int(match.group(1))
                    full_url = urljoin(self.base_url, href)
                    issue_links.append((issue_number, full_url))

            sorted_links = sorted(issue_links, key=lambda x: x[0], reverse=True)
            return [url for _, url in sorted_links]
        except Exception as e:
            print(f"Error getting article links: {e}")
            return []

    def _extract_news_content(self, soup):
        news_header = soup.find('h1', id='news')
        if not news_header:
            print("News section not found")
            return []

        print(f"Found news header: {news_header}")

        news_section = news_header.find_parent()
        if not news_section:
            news_section = soup.find('div', id='content')

        all_elements = list(news_section.children)
        news_start_index = -1

        for i, element in enumerate(all_elements):
            if hasattr(element, 'get') and element.get('id') == 'news':
                news_start_index = i
                break

        if news_start_index == -1:
            print("Could not find news section start")
            return []

        news_elements = []
        for element in all_elements[news_start_index + 1:]:
            if hasattr(element, 'name') and element.name:
                news_elements.append(element)

        print(f"Found {len(news_elements)} elements after news header")

        news_articles = []
        current_article = None
        current_image = None

        i = 0
        while i < len(news_elements):
            element = news_elements[i]

            if element.name == 'figure':
                img = element.find('img')
                if img and img.get('src'):
                    current_image = {
                        'url': urljoin(self.base_url, img['src']),
                        'alt': img.get('alt', ''),
                        'caption': ''
                    }
                    figcaption = element.find('figcaption')
                    if figcaption:
                        current_image['caption'] = figcaption.get_text().strip()

                    print(f"Found image for next article: {current_image['url']}")

            elif element.name == 'h1' and element.get('id'):
                if current_article:
                    news_articles.append(current_article)
                    print(f"Completed article: {current_article['title']} with {len(current_article['raw_content'])} content elements")

                current_article = {
                    'title': element.get_text().strip(),
                    'id': element.get('id'),
                    'image': current_image,
                    'chunks': [],
                    'raw_content': []
                }

                print(f"Started new article: {current_article['title']} (id: {current_article['id']})")
                current_image = None

            elif current_article and element.name in ['p', 'ul', 'ol', 'div']:
                self._process_content_element(element, current_article)

            i += 1

        if current_article:
            news_articles.append(current_article)
            print(f"Completed final article: {current_article['title']} with {len(current_article['raw_content'])} content elements")

        for article in news_articles:
            article['chunks'] = self._create_article_chunks(article['raw_content'])
            del article['raw_content']

        print(f"Total articles extracted: {len(news_articles)}")
        return news_articles

    def _process_content_element(self, element, article):
        if element.name == 'p':
            full_text = element.get_text().strip()
            if full_text:
                article['raw_content'].append({
                    'type': 'paragraph',
                    'content': full_text
                })

        elif element.name in ['ul', 'ol']:
            list_items = []
            for li in element.find_all('li'):
                li_text = li.get_text().strip()
                if li_text:
                    list_items.append(li_text)

            if list_items:
                list_text = "\n".join([f"{i+1}. {item}" for i, item in enumerate(list_items)])
                article['raw_content'].append({
                    'type': 'list',
                    'content': list_text
                })

        elif element.name == 'div':
            text = element.get_text().strip()
            if text and len(text) > 20:
                article['raw_content'].append({
                    'type': 'div_text',
                    'content': text
                })

    def _create_article_chunks(self, raw_content):
        if not raw_content:
            return []

        full_text_parts = [content_item['content'] for content_item in raw_content]
        full_text = "\n\n".join(full_text_parts)

        if not full_text.strip():
            return []

        chunks = self.text_splitter.split_text(full_text)
        return [chunk.strip() for chunk in chunks if chunk.strip()]

    def _parse_date(self, date_string):
        try:
            date_string = date_string.strip()

            for fmt in ["%b %d, %Y", "%b %d %Y"]:
                try:
                    return datetime.strptime(date_string, fmt)
                except ValueError:
                    continue

            print(f"Could not parse date: {date_string}")
            return None
        except Exception as e:
            print(f"Error parsing date {date_string}: {e}")
            return None

    def _extract_issue_id(self, url):
        match = re.search(r'/issue-(\d+)/?', url)
        return int(match.group(1)) if match else None

    def _extract_title(self, soup):
        selectors = [
            '#embed-player > div > div.content-container > div.title-line-container > p > span',
            'div#root span.title',
        ]

        for selector in selectors:
            title_elem = soup.select_one(selector)
            if title_elem:
                title = title_elem.get_text().strip()
                return title.rstrip(" and more...").strip()

        meta_title = soup.find('meta', property='og:title')
        if meta_title and meta_title.get('content'):
            title = meta_title['content'].strip()
            return title.rstrip(" and more...").strip()

        return "No title found"

    def scrape_article(self, url):
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.content, 'html.parser')

            issue_id = self._extract_issue_id(url)

            date_div = soup.select_one('div.mt-1.text-slate-600.text-base.text-sm')
            date_string = date_div.get_text().strip() if date_div else "No date found"

            parsed_date = None
            if date_string != "No date found":
                parsed_date = self._parse_date(date_string)

            main_title = self._extract_title(soup)

            news_articles = self._extract_news_content(soup)

            total_chunks = sum(len(article['chunks']) for article in news_articles)
            total_content_length = sum(
                sum(len(chunk) for chunk in article['chunks'])
                for article in news_articles
            )

            return {
                'url': url,
                'issue_id': issue_id,
                'main_title': main_title,
                'date': parsed_date.isoformat() if parsed_date else None,
                'date_original': date_string,
                'news_articles': news_articles,
                'total_news_count': len(news_articles),
                'chunk_stats': {
                    'total_chunks': total_chunks,
                    'chunk_size_limit': self.chunk_size,
                    'chunk_overlap': self.chunk_overlap,
                    'total_content_length': total_content_length,
                    'avg_chunk_length': total_content_length / total_chunks if total_chunks > 0 else 0,
                    'avg_chunks_per_article': total_chunks / len(news_articles) if news_articles else 0
                },
                'scraped_at': datetime.now().isoformat()
            }

        except Exception as e:
            print(f"Error scraping {url}: {e}")
            return None

    def collect_data(self, max_articles=10):
        print("Getting article links...")
        all_links = self.get_article_links()
        links = all_links[:max_articles]

        print(f"Found {len(links)} articles to scrape")

        for i, link in enumerate(links, 1):
            print(f"Scraping article {i}/{len(links)}: {link}")
            article = self.scrape_article(link)
            if article:
                self.articles.append(article)
                print(f"  - Issue ID: {article['issue_id']}")
                print(f"  - Date: {article['date']}")
                print(f"  - Found {article['total_news_count']} news articles")
                print(f"  - Created {article['chunk_stats']['total_chunks']} total chunks")
                print(f"  - Avg chunk length: {article['chunk_stats']['avg_chunk_length']:.0f} chars")
            time.sleep(1)

        return self.articles

    def _ensure_serializable_date(self, article):
        article_copy = article.copy()
        if article_copy.get('date') and hasattr(article_copy['date'], 'isoformat'):
            article_copy['date'] = article_copy['date'].isoformat()
        return article_copy

    def save_data(self, filename='data/raw/batch_articles.json'):
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        serializable_articles = [self._ensure_serializable_date(article) for article in self.articles]

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(serializable_articles, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(self.articles)} articles to {filename}")

    def save_chunks_only(self, filename='data/processed/batch_chunks.json'):
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        all_chunks = []
        chunk_id = 0

        for article in self.articles:
            for news_article in article.get('news_articles', []):
                for i, chunk in enumerate(news_article.get('chunks', [])):
                    chunk_data = {
                        'chunk_id': chunk_id,
                        'issue_id': article['issue_id'],
                        'issue_url': article['url'],
                        'issue_title': article['main_title'],
                        'issue_date': article['date'],
                        'news_title': news_article['title'],
                        'news_id': news_article['id'],
                        'chunk_index': i,
                        'total_chunks_in_news': len(news_article.get('chunks', [])),
                        'content': chunk,
                        'chunk_length': len(chunk),
                        'image': news_article.get('image')
                    }
                    all_chunks.append(chunk_data)
                    chunk_id += 1

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(all_chunks, f, ensure_ascii=False, indent=2)

        print(f"Saved {len(all_chunks)} chunks to {filename}")
        return all_chunks

    def save_news_articles(self, filename='data/processed/news_articles.json'):
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        all_news = []
        news_id = 0

        for article in self.articles:
            for news_article in article.get('news_articles', []):
                news_data = {
                    'news_id': news_id,
                    'issue_id': article['issue_id'],
                    'issue_url': article['url'],
                    'issue_title': article['main_title'],
                    'issue_date': article['date'],
                    'title': news_article['title'],
                    'id': news_article['id'],
                    'image': news_article.get('image'),
                    'chunks': news_article['chunks'],
                    'total_chunks': len(news_article['chunks']),
                    'content_length': sum(len(chunk) for chunk in news_article['chunks'])
                }
                all_news.append(news_data)
                news_id += 1

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(all_news, f, ensure_ascii=False, indent=2)

        print(f"Saved {len(all_news)} news articles to {filename}")
        return all_news

    def get_chunking_stats(self):
        if not self.articles:
            return "No articles collected yet"

        total_chunks = 0
        total_content_length = 0
        total_news = 0
        chunk_lengths = []

        for article in self.articles:
            for news_article in article.get('news_articles', []):
                total_news += 1
                chunks = news_article.get('chunks', [])
                total_chunks += len(chunks)

                for chunk in chunks:
                    chunk_lengths.append(len(chunk))
                    total_content_length += len(chunk)

        return {
            'total_issues': len(self.articles),
            'total_news_articles': total_news,
            'total_chunks': total_chunks,
            'avg_chunks_per_news': total_chunks / total_news if total_news > 0 else 0,
            'total_content_length': total_content_length,
            'avg_chunk_length': sum(chunk_lengths) / len(chunk_lengths) if chunk_lengths else 0,
            'min_chunk_length': min(chunk_lengths) if chunk_lengths else 0,
            'max_chunk_length': max(chunk_lengths) if chunk_lengths else 0,
            'chunks_over_limit': sum(1 for length in chunk_lengths if length > self.chunk_size),
            'chunk_size_limit': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'chunks_distribution': {
                '0-200': sum(1 for l in chunk_lengths if 0 <= l <= 200),
                '201-500': sum(1 for l in chunk_lengths if 201 <= l <= 500),
                '501-800': sum(1 for l in chunk_lengths if 501 <= l <= 800),
                '801-1000': sum(1 for l in chunk_lengths if 801 <= l <= 1000),
                '1000+': sum(1 for l in chunk_lengths if l > 1000)
            }
        }


if __name__ == "__main__":
    collector = BatchDataCollector()

    articles = collector.collect_data(max_articles=5)
    collector.save_data()
    collector.save_chunks_only()
    collector.save_news_articles()

    stats = collector.get_chunking_stats()
    print(json.dumps(stats, indent=2))