import bs4 as bs
import urllib.request

sauce = urllib.request.urlopen('http://www.data.jma.go.jp/sakura/data/sakura003_03.html').read()
soup = bs.BeautifulSoup(sauce, 'lxml')

for paragraph in soup.find_all('p'):
    print(paragraph.text)