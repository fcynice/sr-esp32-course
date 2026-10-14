# -*- coding: utf-8 -*-
import requests
import re
import os
import time
import random
from pathlib import Path

# 保存到脚本所在目录下的 data 文件夹
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"

image_src = 'https://image.baidu.com/search/acjson'
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36 Edg/125.0.0.0',
    'Referer': 'https://image.baidu.com/',
}
params = {
    'tn': 'resultjson_com',
    'logid': '7008269991081857061',
    'ipn': 'rj',
    'ct': '201326592',
    'is': '',
    'fp': 'result',
    'fr': '',
    'word': '',
    'queryWord': '',
    'cl': '2',
    'lm': '',
    'ie': 'utf-8',
    'oe': 'utf-8',
    'adpicid': '',
    'st': '-1',
    'z': '',
    'ic': '0',
    'hd': '',
    'latest': '',
    'copyright': '',
    's': '',
    'se': '',
    'tab': '',
    'width': '',
    'height': '',
    'face': '0',
    'istype': '2',
    'qc': '',
    'nc': '1',
    'expermode': '',
    'nojc': '',
    'isAsync': '',
    'pn': '',
    'rn': '30',
    '1716190281723': ''
}

image_name = input('请输入需要爬取的图片关键字：')
image_count = int(input('请输入爬取图片的数量：'))
params['word'] = image_name
params['queryWord'] = image_name

save_dir = DATA_DIR / image_name


def image():
    count = 1
    in_count = 0
    if save_dir.exists():
        in_count = len(os.listdir(save_dir))
        count = int(in_count / 30) + 3

    empty_page_count = 0  # 连续空页计数器

    while True:
        params['pn'] = count * 30
        try:
            resp = requests.get(url=image_src, headers=headers, params=params, timeout=15)
            data = resp.json()
        except Exception as e:
            print(f'⚠️ 请求异常: {e}，跳过本页')
            count += 1
            time.sleep(random.uniform(3, 6))
            continue

        items = [it for it in data.get('data', []) if isinstance(it, dict)]
        if not items:
            empty_page_count += 1
            print(f'⚠️ 第 {count} 页返回空数据（连续空页 {empty_page_count}/3）')
            if empty_page_count >= 3:
                print('❌ 连续 3 页为空，停止爬取。')
                break
            count += 1
            time.sleep(random.uniform(3, 6))
            continue

        empty_page_count = 0  # 有数据就重置

        for url in items[:30]:
            title = re.sub(r'[\\/:*?"<>| ]', '', url.get('fromPageTitle', f'img_{count}'))
            if not title:
                title = f'img_{count}'

            # 关键：优先 middleURL（百度CDN，稳定，500~800像素）
            img_link = url.get('middleURL') 
            # or url.get('thumbURL') 兜底 thumbURL
            if not img_link:
                continue

            try:
                img_data = requests.get(url=img_link, headers=headers, timeout=15).content
            except Exception:
                continue

            if len(img_data) < 5120:  # 过滤小于5KB的无效图
                continue

            if not save_dir.exists():
                save_dir.mkdir(parents=True, exist_ok=True)

            filepath = save_dir / f'{title}_{in_count + 1}.png'
            with open(filepath, 'wb') as f:
                f.write(img_data)

            in_count += 1
            print(f'正在下载第{in_count}张图片')

            if in_count >= image_count:
                print('下载完成')
                exit()

            time.sleep(random.uniform(0.3, 0.8))

        count += 1
        time.sleep(random.uniform(2, 4))


if __name__ == '__main__':
    image()