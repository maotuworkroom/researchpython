from googlesearch import search
import requests
from fake_useragent import UserAgent

def search_bing(keywords):
    # 使用 htmlapi.xinu.ink 获取 Bing 搜索结果
    try:
        response = requests.get(f'https://htmlapi.xinu.ink/api/extract?url=https://www.bing.com/search?q={keywords}&output_format=html')
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                # 解析 HTML 并提取链接
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(data.get('content', ''), 'html.parser')
                links = []
                for item in soup.find_all('a'):
                    href = item.get('href')
                    if href and href.startswith(('http://', 'https://')):
                        links.append(href)
                return links
            else:
                print(f"Failed to extract content using htmlapi.xinu.ink: {data}")
        else:
            print(f"htmlapi.xinu.ink request failed with status code: {response.status_code}")
    except Exception as e:
        print(f"Error using htmlapi.xinu.ink: {e}")
    return []

def search_duckduckgo(keywords):
    # 使用 DuckDuckGo API 进行搜索
    response = requests.get(f'https://api.duckduckgo.com/?q={keywords}&format=json')
    return [item['FirstURL'] for item in response.json().get('RelatedTopics', [])]

def search_baidu(keywords):
    # 模拟百度搜索
    headers = {'User-Agent': UserAgent().random}
    response = requests.get(f'https://www.baidu.com/s?wd={keywords}', headers=headers)
    # 解析 HTML 并提取链接
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(response.text, 'html.parser')
    links = []
    for item in soup.find_all('a'):
        href = item.get('href')
        if href and href.startswith(('http://', 'https://')):
            links.append(href)
    return links

def search_yandex(keywords):
    # 模拟 Yandex 搜索
    headers = {'User-Agent': UserAgent().random}
    response = requests.get(f'https://yandex.com/search/?text={keywords}', headers=headers)
    # 解析 HTML 并提取链接
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(response.text, 'html.parser')
    links = []
    for item in soup.find_all('a'):
        href = item.get('href')
        if href and href.startswith(('http://', 'https://')):
            links.append(href)
    return links

def search_chaoxing(keywords):
    # 超星期刊搜索
    try:
        response = requests.get(f'https://htmlapi.xinu.ink/api/extract?url=https://qikan.chaoxing.com/searchjour?sw={keywords}&size=50&output_format=html')
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(data.get('content', ''), 'html.parser')
                links = []
                for item in soup.find_all('a'):
                    href = item.get('href')
                    if href and href.startswith(('http://', 'https://')):
                        links.append(href)
                return links
            else:
                print(f"Failed to extract content from Chaoxing: {data}")
        else:
            print(f"Chaoxing request failed with status code: {response.status_code}")
    except Exception as e:
        print(f"Error searching Chaoxing: {e}")
    return []

def search_arxiv(keywords):
    # arxiv搜索
    try:
        response = requests.get(f'https://htmlapi.xinu.ink/api/extract?url=https://arxiv.org/search/?query={keywords}&searchtype=all&abstracts=show&order=-announced_date_first&size=50&output_format=html')
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(data.get('content', ''), 'html.parser')
                links = []
                for item in soup.find_all('a'):
                    href = item.get('href')
                    if href and href.startswith(('http://', 'https://')):
                        links.append(href)
                return links
            else:
                print(f"Failed to extract content from arXiv: {data}")
        else:
            print(f"arXiv request failed with status code: {response.status_code}")
    except Exception as e:
        print(f"Error searching arXiv: {e}")
    return []

def search_weipu(keywords):
    # 维普搜索
    try:
        response = requests.get(f'https://htmlapi.xinu.ink/api/extract?url=https://mqikan.cqvip.com/Article/index?from=Article_index&key=U%3D{keywords}&output_format=html')
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(data.get('content', ''), 'html.parser')
                links = []
                for item in soup.find_all('a'):
                    href = item.get('href')
                    if href and href.startswith(('http://', 'https://')):
                        links.append(href)
                return links
            else:
                print(f"Failed to extract content from Weipu: {data}")
        else:
            print(f"Weipu request failed with status code: {response.status_code}")
    except Exception as e:
        print(f"Error searching Weipu: {e}")
    return []

def search_semantic_scholar(keywords):
    # Semantic Scholar搜索
    try:
        response = requests.get(f'https://htmlapi.xinu.ink/api/extract?url=https://www.semanticscholar.org/search?q={keywords}&sort=relevance&output_format=html')
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(data.get('content', ''), 'html.parser')
                links = []
                for item in soup.find_all('a'):
                    href = item.get('href')
                    if href and href.startswith(('http://', 'https://')):
                        links.append(href)
                return links
            else:
                print(f"Failed to extract content from Semantic Scholar: {data}")
        else:
            print(f"Semantic Scholar request failed with status code: {response.status_code}")
    except Exception as e:
        print(f"Error searching Semantic Scholar: {e}")
    return []

def search_annas_archive(keywords):
    # 安妮档案搜索
    try:
        response = requests.get(f'https://htmlapi.xinu.ink/api/extract?url=https://zh.annas-archive.org/search?index=journals&q={keywords}&output_format=html')
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(data.get('content', ''), 'html.parser')
                links = []
                for item in soup.find_all('a'):
                    href = item.get('href')
                    if href and href.startswith(('http://', 'https://')):
                        links.append(href)
                return links
            else:
                print(f"Failed to extract content from Anna's Archive: {data}")
        else:
            print(f"Anna's Archive request failed with status code: {response.status_code}")
    except Exception as e:
        print(f"Error searching Anna's Archive: {e}")
    return []

def get_links(keywords, academic=False):
    try:
        results = []
        
        if academic:
            # 超星期刊
            try:
                results.extend(search_chaoxing(keywords))
                if results:
                    return results
            except Exception as e:
                print(f"[INFO] Chaoxing search failed: {e}")

            # arxiv
            try:
                results.extend(search_arxiv(keywords))
                if results:
                    return results
            except Exception as e:
                print(f"[INFO] arXiv search failed: {e}")

            # 维普
            try:
                results.extend(search_weipu(keywords))
                if results:
                    return results
            except Exception as e:
                print(f"[INFO] Weipu search failed: {e}")

            # Semantic Scholar
            try:
                results.extend(search_semantic_scholar(keywords))
                if results:
                    return results
            except Exception as e:
                print(f"[INFO] Semantic Scholar search failed: {e}")

            # 安妮档案
            try:
                results.extend(search_annas_archive(keywords))
                if results:
                    return results
            except Exception as e:
                print(f"[INFO] Anna's Archive search failed: {e}")

        else:
            # Google Search
            try:
                for result in search(keywords):
                    if result.startswith(('http://', 'https://')):
                        results.append(result)
                if results:
                    return results
            except Exception as e:
                print(f"[INFO] Google search failed: {e}")

            # Baidu Search
            try:
                results.extend(search_baidu(keywords))
                if results:
                    return results
            except Exception as e:
                print(f"[INFO] Baidu search failed: {e}")

            # DuckDuckGo Search
            try:
                results.extend(search_duckduckgo(keywords))
                if results:
                    return results
            except Exception as e:
                print(f"[INFO] DuckDuckGo search failed: {e}")

            # Bing Search
            try:
                results.extend(search_bing(keywords))
                if results:
                    return results
            except Exception as e:
                print(f"[INFO] Bing search failed: {e}")

            # Yandex Search
            try:
                results.extend(search_yandex(keywords))
                if results:
                    return results
            except Exception as e:
                print(f"[INFO] Yandex search failed: {e}")

        return results
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
            print(f"[INFO] Too many requests.")
            return {'result': 429}
        raise
    except requests.exceptions.ReadTimeout as e:
        print(f"[ERROR] Request timed out while searching for keywords: {keywords}")
        return {'result': 'timeout'}

from newspaper import Article, ArticleException, Config

config = Config()

def get_text(url):
    config.browser_user_agent = UserAgent().random
    article = Article(url, config=config)
    try:
        article.download()
        article.parse()
        return article.title, article.text
    except ArticleException as e:
        print(f"Failed to download article from {url}: {e}")
    except FileNotFoundError as e:
        print(f"Directory not found for article resources: {e}")

    # 使用备份方案
    try:
        response = requests.get(f'https://htmlapi.xinu.ink/api/extract?url={url}&output_format=markdown')
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                # 提取标题和内容
                title = data.get('url', 'No Title')
                content = data.get('content', '')
                return title, content
            else:
                print(f"Failed to extract content using backup tool: {data}")
        else:
            print(f"Backup tool request failed with status code: {response.status_code}")
    except Exception as e:
        print(f"Error using backup tool: {e}")

    return "Error: Unable to retrieve article.", ""

from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import urllib.parse
import json

class RequestHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        
        if self.path.startswith('/get_links'):
            parsed_path = urllib.parse.urlparse(self.path)
            query_params = urllib.parse.parse_qs(parsed_path.query)
            
            keywords = query_params.get('keywords', [''])[0]
            links = get_links(keywords)
            
            if isinstance(links, dict) and 'result' in links and links['result'] == 429:
                self.wfile.write(json.dumps(links).encode())
                return
            
            print('Found links: ', links)            
            self.wfile.write(json.dumps({'result': links}).encode())

        elif self.path.startswith('/get_link_texts'):
            parsed_path = urllib.parse.urlparse(self.path)
            query_params = urllib.parse.parse_qs(parsed_path.query)

            links_param = query_params.get('links', [''])[0]
            links = json.loads(links_param)
            print(f"Received {len(links)} links: {links}")

            texts = []
            for link in links:
                print(f"Reading url: {link}")
                article_title, link_text = get_text(link)

                if len(link_text) < 300 or article_title.split(' ')[0] == 'Error:': 
                    texts.append({
                        'url': link,
                        'length': 0
                    })
                else:           
                    texts.append({
                        'url': link,
                        'length': len(link_text),
                        'text': link_text,
                        'title': article_title
                    })

            self.wfile.write(json.dumps({'results': texts}).encode())

        # 新增测试端点
        elif self.path.startswith('/test_bing'):
            parsed_path = urllib.parse.urlparse(self.path)
            query_params = urllib.parse.parse_qs(parsed_path.query)
            keywords = query_params.get('keywords', [''])[0]
            links = search_bing(keywords)
            self.wfile.write(json.dumps({'result': links}).encode())

        elif self.path.startswith('/test_nw'):
            parsed_path = urllib.parse.urlparse(self.path)
            query_params = urllib.parse.parse_qs(parsed_path.query)
            keywords = query_params.get('keywords', [''])[0]
            links = search_duckduckgo(keywords)
            self.wfile.write(json.dumps({'result': links}).encode())

        elif self.path.startswith('/test_xinu'):
            parsed_path = urllib.parse.urlparse(self.path)
            query_params = urllib.parse.parse_qs(parsed_path.query)
            keywords = query_params.get('keywords', [''])[0]
            links = search_baidu(keywords)
            self.wfile.write(json.dumps({'result': links}).encode())

        else:
            super().do_GET()


if __name__ == '__main__':
    server_address = ('', 8000)
    httpd = ThreadingHTTPServer(server_address, RequestHandler)
    print('Server running on port 8000')
    httpd.serve_forever()
