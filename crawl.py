#!/usr/bin/env python3
"""
必应图片爬虫 — 用于超分辨率训练数据采集
按关键词搜索，按最小分辨率筛选，批量下载高清图片。
"""

import argparse
import html
import json
import re
import time
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm


BING_ASYNC_URL = "https://www.bing.com/images/async"

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Referer": "https://www.bing.com/images/search",
}

VALID_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff")


def build_session():
    """创建带自动重试的 requests Session"""
    session = requests.Session()
    session.headers.update(DEFAULT_HEADERS)
    retry = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD"],
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def parse_bing_page(html_text):
    """
    从必应异步接口返回的 HTML 中提取图片元数据。
    每个结果形如 <a class="iusc" m='{"murl":"...","mw":1920,"mh":1080}' ...>
    """
    results = []
    # 以 <a 标签为界切块，避免跨标签误匹配
    chunks = re.split(r"(?=<a\s)", html_text)

    for chunk in chunks:
        if 'class="iusc"' not in chunk:
            continue
        # m 属性可能用单引号或双引号包裹
        m = re.search(r"\bm='([^']*)'", chunk)
        if not m:
            m = re.search(r'\bm="([^"]*)"', chunk)
        if not m:
            continue

        raw = html.unescape(m.group(1))
        try:
            meta = json.loads(raw)
        except json.JSONDecodeError:
            continue

        murl = meta.get("murl") or meta.get("m")
        if not murl or not isinstance(murl, str) or not murl.startswith("http"):
            continue

        results.append({
            "url": murl,
            "width": int(meta.get("mw") or 0),
            "height": int(meta.get("mh") or 0),
        })
    return results


def search_images(session, keyword, pages, per_page, min_width, min_height, delay):
    """按关键词分页搜索，返回过滤后的图片元数据列表"""
    collected = []
    seen = set()

    for page in range(pages):
        first = 1 + page * per_page
        params = {
            "q": keyword,
            "first": first,
            "count": per_page,
            "mmasync": 1,
        }
        try:
            resp = session.get(BING_ASYNC_URL, params=params, timeout=15)
            resp.raise_for_status()
        except requests.RequestException as e:
            print(f"[!] 第 {page + 1} 页请求失败: {e}")
            time.sleep(delay * 2)
            continue

        items = parse_bing_page(resp.text)
        if not items:
            print(f"[i] 第 {page + 1} 页没有解析到结果，可能已到末尾")
            break

        new = 0
        for item in items:
            if item["url"] in seen:
                continue
            if item["width"] >= min_width and item["height"] >= min_height:
                seen.add(item["url"])
                collected.append(item)
                new += 1

        print(f"[+] 第 {page + 1} 页: 解析 {len(items)} 条, "
              f"新增 {new} 张, 累计符合要求 {len(collected)} 张")
        time.sleep(delay)

    return collected


def download_images(session, items, output_dir, delay, timeout):
    """下载图片到指定目录"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    success, failed = 0, 0
    for i, item in enumerate(tqdm(items, desc="下载中", unit="张")):
        url = item["url"]
        ext = Path(url.split("?")[0]).suffix.lower()
        if ext not in VALID_EXTS:
            ext = ".jpg"
        filename = output_dir / f"{i:05d}_{item['width']}x{item['height']}{ext}"

        if filename.exists():
            continue

        try:
            r = session.get(url, timeout=timeout, stream=True)
            r.raise_for_status()
            content_type = r.headers.get("Content-Type", "")
            if "image" not in content_type and not ext:
                raise ValueError(f"非图片内容: {content_type}")
            with open(filename, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            success += 1
        except Exception as e:
            failed += 1
            tqdm.write(f"[!] 下载失败 ({i}): {e}")
            if filename.exists():
                filename.unlink()
        time.sleep(delay)

    return success, failed


def main():
    parser = argparse.ArgumentParser(
        description="必应图片爬虫 — 按关键词和分辨率采集超分训练数据"
    )
    parser.add_argument("-k", "--keyword", required=True,
                        help="搜索关键词，如 '中国大学校园 日常'")
    parser.add_argument("-o", "--output", default="./dataset",
                        help="输出目录 (默认 ./dataset)")
    parser.add_argument("-n", "--num", type=int, default=200,
                        help="目标下载数量 (默认 200)")
    parser.add_argument("--pages", type=int, default=20,
                        help="最多搜索页数 (默认 20)")
    parser.add_argument("--per-page", type=int, default=35,
                        help="每页条数 (默认 35)")
    parser.add_argument("--min-width", type=int, default=1024,
                        help="最小宽度 px (默认 1024)")
    parser.add_argument("--min-height", type=int, default=1024,
                        help="最小高度 px (默认 1024)")
    parser.add_argument("--delay", type=float, default=1.0,
                        help="搜索请求间隔秒数 (默认 1.0)")
    parser.add_argument("--dl-delay", type=float, default=0.5,
                        help="下载请求间隔秒数 (默认 0.5)")
    args = parser.parse_args()

    session = build_session()

    print(f"[*] 关键词: {args.keyword}")
    print(f"[*] 分辨率要求: >= {args.min_width} x {args.min_height}")
    print(f"[*] 目标数量: {args.num}")

    items = search_images(
        session,
        keyword=args.keyword,
        pages=args.pages,
        per_page=args.per_page,
        min_width=args.min_width,
        min_height=args.min_height,
        delay=args.delay,
    )

    if not items:
        print("[!] 未获取到任何图片，请检查关键词或网络")
        return

    items = items[: args.num]
    print(f"\n[*] 准备下载 {len(items)} 张到 {args.output}\n")

    ok, bad = download_images(
        session, items, args.output,
        delay=args.dl_delay, timeout=20,
    )
    print(f"\n[✓] 完成: 成功 {ok} 张, 失败 {bad} 张")
    print(f"[✓] 保存位置: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()