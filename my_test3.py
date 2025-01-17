# Author：Kevin Sun
# 上海股票信息

import requests
from bs4 import BeautifulSoup
import traceback
import re
import time


def getHTMLText(url):  # 获得所需的网页源代码
    try:
        user_agent = 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.87 Safari/537.36'
        headers = {'User-Agent': user_agent}
        r = requests.get(url, headers=headers, timeout=30)
        r.raise_for_status()
        r.encoding = r.apparent_encoding
        return r.text
    except:
        return ""


def getFileName():
    dirname = time.strftime('%Y%m%d', time.localtime(time.time()))
    dirname += 'sh'
    return dirname


def getStockList(lst, stock_list_url):  # 获得东方财富网上面上海股票的全部代码
    html = getHTMLText(stock_list_url)
    soup = BeautifulSoup(html, 'html.parser')
    a = soup.find_all('a')  # 用find_all方法遍历所有'a'标签，然后在'a'标签里面提取出href部分信息
    for i in a:
        try:
            href = i.attrs['href']
            lst.append(re.findall(r"sh6\d{5}", href)[0])  # 用正则表达式匹配所需的信息，“sh\d{6}”
            # print(lst)
        except:
            continue


def getStockInfo(lst, stock_info_url, fpath):
    ndate = time.strftime('%Y%m%d', time.localtime(time.time()))
    for stock in lst:
        url = stock_info_url + stock + '.html'  # 拼接url
        html = getHTMLText(url)
        try:
            if html == "":
                continue
            infoDict = {}
            soup = BeautifulSoup(html, 'html.parser')
            stockInfo = soup.find('div', attrs={'class': 'stock-bets'})
            if stockInfo == None:  # 判断为空，返回
                continue
            # print(stockInfo)
            # name = stockInfo.find_all(attrs={'class': 'bets-name'})[0]
            # print(name)
            # infoDict.update({'股票编码':stock})
            # inp=name.text.split()[0]+":"
            keyList = stockInfo.find_all('dt')
            valueList = stockInfo.find_all('dd')
            inp = stock + ndate + "," + stock + "," + ndate + ","
            for i in range(len(keyList)):
                key = keyList[i].text
                val = valueList[i].text
                infoDict[key] = val
            # print(inp)
            inp += infoDict['最高'] + "," + infoDict['换手率'] + "," + infoDict['成交量'] + "," + infoDict['成交额'] + "\n"
            print(inp)
            with open(fpath, 'a', encoding='utf-8') as f:

                # f.write(str(infoDict) + '\n')
                f.write(inp)
        except:
            traceback.print_exc()
            continue


def main():  # 主方法调用上面的函数
    stock_list_url = 'http://quote.eastmoney.com/stocklist.html'
    stock_info_url = 'http://gupiao.baidu.com/stock/'
    output_file = './' + getFileName() + '.txt'
    slist = []
    getStockList(slist, stock_list_url)
    getStockInfo(slist, stock_info_url, output_file)


main()