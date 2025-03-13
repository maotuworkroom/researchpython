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

def search_academic(keywords):
    results = []

    # 百度学术搜索
    try:
        headers = {'User-Agent': UserAgent().random}
        response = requests.get(f'https://xueshu.baidu.com/s?wd={keywords}', headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        for item in soup.find_all('a'):
            href = item.get('href')
            if href and href.startswith(('http://', 'https://')):
                results.append(href)
    except Exception as e:
        print(f"[INFO] Baidu Academic search failed: {e}")

    # 必应学术搜索
    try:
        headers = {'User-Agent': UserAgent().random}
        response = requests.get(f'https://www.bing.com/academic?q={keywords}', headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        for item in soup.find_all('a'):
            href = item.get('href')
            if href and href.startswith(('http://', 'https://')):
                results.append(href)
    except Exception as e:
        print(f"[INFO] Bing Academic search failed: {e}")

    # 谷歌学术搜索
    try:
        headers = {'User-Agent': UserAgent().random}
        response = requests.get(f'https://scholar.google.com/scholar?q={keywords}', headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        for item in soup.find_all('a'):
            href = item.get('href')
            if href and href.startswith(('http://', 'https://')):
                results.append(href)
    except Exception as e:
        print(f"[INFO] Google Scholar search failed: {e}")

    return results

def get_links(keywords, academic=False):
    try:
        results = []
        
        if academic:
            # 使用学术搜索
            results.extend(search_academic(keywords))
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

# 移除 timeout_decorator 导入
from PyPDF2 import PdfReader
import io
import docx

# 修改 get_text 函数，移除 timeout_decorator 装饰器
def get_text(url):
    config.browser_user_agent = UserAgent().random
    headers = {'User-Agent': UserAgent().random}
    
    try:
        MAX_CONTENT_LENGTH = 5 * 1024 * 1024  # 5MB 限制
        
        # 优先使用 htmlapi.xinu.ink
        try:
            response = requests.get(
                f'https://htmlapi.xinu.ink/api/extract?url={url}&output_format=markdown',
                timeout=10,
                stream=True
            )
            content_length = int(response.headers.get('content-length', 0))
            if content_length > MAX_CONTENT_LENGTH:
                return "Error: Content too large", ""
                
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    content = data.get('content', '')
                    if len(content) > MAX_CONTENT_LENGTH:
                        content = content[:MAX_CONTENT_LENGTH] + "..."
                    return data.get('url', 'No Title'), content
        except requests.Timeout:
            print(f"Primary API timeout for {url}")
        except Exception as e:
            print(f"Primary API failed for {url}: {e}")

        # PDF 处理优化
        if url.lower().endswith('.pdf'):
            response = requests.get(url, headers=headers, timeout=10, stream=True)
            content_length = int(response.headers.get('content-length', 0))
            if content_length > MAX_CONTENT_LENGTH:
                return "Error: PDF too large", ""
                
            pdf = PdfReader(io.BytesIO(response.content))
            text = ""
            # 限制处理页数
            max_pages = min(len(pdf.pages), 20)  
            for i in range(max_pages):
                text += pdf.pages[i].extract_text() + "\n"
                if len(text) > MAX_CONTENT_LENGTH:
                    text = text[:MAX_CONTENT_LENGTH] + "..."
                    break
            return url, text
            
        elif url.lower().endswith('.docx'):
            response = requests.get(url, headers=headers, timeout=10, stream=True)
            content_length = int(response.headers.get('content-length', 0))
            if content_length > MAX_CONTENT_LENGTH:
                return "Error: DOCX too large", ""
                
            doc = docx.Document(io.BytesIO(response.content))
            text = ""
            for para in doc.paragraphs:
                text += para.text + "\n"
                if len(text) > MAX_CONTENT_LENGTH:
                    text = text[:MAX_CONTENT_LENGTH] + "..."
                    break
            return url, text

        # newspaper3k 优化
        try:
            article = Article(url, config=config)
            article.download()
            article.parse()
            text = article.text
            if len(text) > MAX_CONTENT_LENGTH:
                text = text[:MAX_CONTENT_LENGTH] + "..."
            if text:
                return article.title, text
        except Exception as e:
            print(f"Newspaper3k failed for {url}: {e}")

        # 直接请求优化
        try:
            response = requests.get(url, headers=headers, timeout=10, stream=True)
            content_length = int(response.headers.get('content-length', 0))
            if content_length > MAX_CONTENT_LENGTH:
                return "Error: Content too large", ""
                
            response.encoding = response.apparent_encoding
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 清理内存
            for tag in soup(['script', 'style', 'nav', 'footer', 'header']):
                tag.decompose()
            
            text = soup.get_text(separator='\n', strip=True)
            if len(text) > MAX_CONTENT_LENGTH:
                text = text[:MAX_CONTENT_LENGTH] + "..."
            if len(text) > 300:
                return url, text
            
            # 清理内存
            soup.decompose()
            
        except requests.Timeout:
            print(f"Request timed out for website: {url}")
            return "Error: Request timed out", ""
        except Exception as e:
            print(f"Error processing {url}: {e}")
            return f"Error: {str(e)}", ""
        finally:
            import gc
            gc.collect()

    return "Error: Unable to retrieve article.", ""

# 修改错误处理部分
class RequestHandler(SimpleHTTPRequestHandler):
    def send_error_json(self, error_message):
        self.send_response(500)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({
            'error': error_message
        }).encode())

    def do_GET(self):
        try:
            self.send_response(200)
            self.send_cors_headers()
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            if self.path.startswith('/get_links'):
                parsed_path = urllib.parse.urlparse(self.path)
                query_params = urllib.parse.parse_qs(parsed_path.query)
                
                keywords = query_params.get('keywords', [''])[0]
                academic = query_params.get('academic', [False])[0]
                
                # 限制同时处理的链接数量
                MAX_LINKS = 10
                links = get_links(keywords, academic)
                if isinstance(links, list):
                    links = links[:MAX_LINKS]
                
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

                MAX_LINKS = 5
                links = links[:MAX_LINKS]
                
                texts = []
                for link in links:
                    try:
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
                    except Exception as e:
                        print(f"Error processing link {link}: {e}")
                        texts.append({
                            'url': link,
                            'length': 0,
                            'error': str(e)
                        })

                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'results': texts}).encode())

            else:
                super().do_GET()
                
        except Exception as e:
            print(f"Error in request handler: {e}")
            self.send_error_json(str(e))
        finally:
            # 强制垃圾回收
            import gc
            gc.collect()


if __name__ == '__main__':
    server_address = ('', 8000)
    httpd = ThreadingHTTPServer(server_address, RequestHandler)
    print('Server running on port 8000')
    httpd.serve_forever()
