"""
Philippine Fake News Dataset Scraper  (v3 — expanded sources)
=============================================================
Scrapes articles from Philippine news sources and saves them to a CSV file.

Fake-labeled sources:
  - VERA Files Fact Check  (verafiles.org/fact-check)
  - Rappler Fact Check     (rappler.com/newsbreak/fact-check)
  - TSEK.PH Fact Check     (tsek.ph)

Real-labeled sources:
  - Inquirer News          (inquirer.net)
  - GMA News               (gmanetwork.com)
  - Manila Bulletin        (mb.com.ph)
  - Philstar               (philstar.com)
  - ABS-CBN News           (news.abs-cbn.com)
  - Philippine News Agency (pna.gov.ph)

Usage:
  python scraper.py
  python scraper.py --fake 200 --real 100   # quick test run
  python scraper.py --fake 1000 --real 500  # large run
"""

import argparse
import csv
import time
import random
import logging
from dataclasses import dataclass, fields

import requests
from bs4 import BeautifulSoup

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OUTPUT_FILE            = "dataset.csv"
DEFAULT_FAKE_PER_SOURCE = 500
DEFAULT_REAL_PER_SOURCE = 250
REQUEST_DELAY          = (1.5, 3.0)
REQUEST_TIMEOUT        = 20

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Article:
    title:          str
    text:           str
    label:          str
    source:         str
    url:            str
    date:           str
    source_dataset: str = "scraped"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get(url: str) -> BeautifulSoup | None:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return BeautifulSoup(resp.text, "html.parser")
    except Exception as exc:
        log.warning(f"  SKIP  {url}  ({exc})")
        return None


def get_rss(url: str) -> list[dict]:
    """Fetch an RSS/Atom feed and return a list of {title, link, date} dicts."""
    import xml.etree.ElementTree as ET
    try:
        resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        items = []
        # RSS 2.0
        for item in root.iter("item"):
            link = (item.findtext("link") or "").strip()
            title = (item.findtext("title") or "").strip()
            date = (item.findtext("pubDate") or "").strip()
            if link and title:
                items.append({"title": title, "link": link, "date": date})
        # Atom
        if not items:
            for entry in root.iter("{http://www.w3.org/2005/Atom}entry"):
                link_el = entry.find("{http://www.w3.org/2005/Atom}link")
                link = (link_el.get("href", "") if link_el is not None else "").strip()
                title = (entry.findtext("{http://www.w3.org/2005/Atom}title") or "").strip()
                date = (entry.findtext("{http://www.w3.org/2005/Atom}published") or "").strip()
                if link and title:
                    items.append({"title": title, "link": link, "date": date})
        return items
    except Exception as exc:
        log.warning(f"  RSS SKIP  {url}  ({exc})")
        return []


def sleep():
    time.sleep(random.uniform(*REQUEST_DELAY))


def clean(text: str) -> str:
    return " ".join(text.split()).strip()


def body_text(soup: BeautifulSoup, selectors: list[str]) -> str:
    for sel in selectors:
        container = soup.select_one(sel)
        if container:
            paras = container.find_all("p")
            text = " ".join(clean(p.get_text()) for p in paras if len(p.get_text()) > 40)
            if len(text) > 200:
                return text
    return ""


# ---------------------------------------------------------------------------
# VERA Files — fake-labeled  (fixed: /fact-check?page=N)
# ---------------------------------------------------------------------------

def scrape_vera_files(limit: int) -> list[Article]:
    log.info("=== VERA Files Fact Check ===")
    articles: list[Article] = []
    seen: set[str] = set()
    page = 1
    max_pages = 60

    while len(articles) < limit and page <= max_pages:
        url = f"https://verafiles.org/fact-check?page={page}"
        log.info(f"  Listing page {page}  ({url})")
        soup = get(url)
        if not soup:
            break

        links = [
            a.get("href", "")
            for a in soup.find_all("a", href=True)
            if "verafiles.org/articles/" in a.get("href", "")
        ]
        if not links:
            log.info("  No article links found — stopping")
            break

        new_links = 0
        for href in links:
            if href in seen:
                continue
            seen.add(href)
            new_links += 1
            article = _fetch_vera_article(href)
            if article:
                articles.append(article)
                log.info(f"  [{len(articles):>3}/{limit}] {article.title[:70]}")
                sleep()
            if len(articles) >= limit:
                break

        if new_links == 0:
            log.info("  No new links — stopping")
            break

        page += 1
        sleep()

    log.info(f"  Collected {len(articles)} articles from VERA Files\n")
    return articles


def _fetch_vera_article(url: str) -> Article | None:
    soup = get(url)
    if not soup:
        return None

    title_tag = soup.select_one("h1.entry-title, h1.article-title, h1")
    title = clean(title_tag.get_text()) if title_tag else ""
    if not title or len(title) < 10:
        return None

    text = body_text(soup, [
        "div.entry-content",
        "div.article-body",
        "div.post-content",
        "article",
    ])
    if len(text) < 150:
        return None

    date_tag = soup.select_one("time, .entry-date, .post-date")
    date = date_tag.get("datetime", date_tag.get_text()) if date_tag else ""

    return Article(title=title, text=text, label="fake",
                   source="VERA Files", url=url, date=clean(date))


# ---------------------------------------------------------------------------
# Rappler Fact Check — fake-labeled  (fixed: /newsbreak/fact-check/page/N/)
# ---------------------------------------------------------------------------

def scrape_rappler(limit: int) -> list[Article]:
    log.info("=== Rappler Fact Check ===")
    articles: list[Article] = []
    seen: set[str] = set()
    page = 1
    max_pages = 80

    while len(articles) < limit and page <= max_pages:
        url = f"https://www.rappler.com/newsbreak/fact-check/page/{page}/"
        log.info(f"  Listing page {page}  ({url})")
        soup = get(url)
        if not soup:
            break

        links = [
            a.get("href", "")
            for a in soup.find_all("a", href=True)
            if "rappler.com/newsbreak/fact-check/" in a.get("href", "")
            and a.get("href", "").rstrip("/") != "https://www.rappler.com/newsbreak/fact-check"
            and len(a.get("href", "").split("/")) > 6
        ]
        if not links:
            log.info("  No listing links found — stopping")
            break

        new_links = 0
        for href in links:
            if href in seen:
                continue
            seen.add(href)
            new_links += 1
            article = _fetch_rappler_article(href)
            if article:
                articles.append(article)
                log.info(f"  [{len(articles):>3}/{limit}] {article.title[:70]}")
                sleep()
            if len(articles) >= limit:
                break

        if new_links == 0:
            log.info("  No new links on this page — stopping")
            break

        page += 1
        sleep()

    log.info(f"  Collected {len(articles)} articles from Rappler\n")
    return articles


def _fetch_rappler_article(url: str) -> Article | None:
    soup = get(url)
    if not soup:
        return None

    title_tag = soup.select_one(
        "h1.article__title, h1.post-title, h1.entry-title, h1"
    )
    title = clean(title_tag.get_text()) if title_tag else ""
    if not title or len(title) < 10:
        return None

    text = body_text(soup, [
        "div.article__content",
        "div.entry-content",
        "div.post-body",
        "article",
    ])
    if len(text) < 150:
        return None

    date_tag = soup.select_one("time, .article__publish-date")
    date = date_tag.get("datetime", date_tag.get_text()) if date_tag else ""

    return Article(title=title, text=text, label="fake",
                   source="Rappler", url=url, date=clean(date))


# ---------------------------------------------------------------------------
# Inquirer — real-labeled  (multiple sections)
# ---------------------------------------------------------------------------

INQUIRER_SECTIONS = [
    "https://newsinfo.inquirer.net/",
    "https://business.inquirer.net/",
    "https://globalnation.inquirer.net/",
    "https://technology.inquirer.net/",
    "https://opinion.inquirer.net/",
    "https://lifestyle.inquirer.net/",
    "https://sports.inquirer.net/",
    "https://entertainment.inquirer.net/",
]


def scrape_inquirer(limit: int) -> list[Article]:
    log.info("=== Inquirer News ===")
    articles: list[Article] = []
    seen: set[str] = set()

    for section_url in INQUIRER_SECTIONS:
        if len(articles) >= limit:
            break
        for page in range(1, 25):
            if len(articles) >= limit:
                break
            paged = section_url if page == 1 else f"{section_url}page/{page}/"
            log.info(f"  Page {page}: {paged}")
            soup = get(paged)
            if not soup:
                break
            links = soup.select("h2 a, h3 a, .article-title a, .headline a, .item-title a, h4 a")
            found = 0
            for a in links:
                href = a.get("href", "")
                if not href or href in seen or "inquirer.net" not in href:
                    continue
                seen.add(href)
                found += 1
                article = _fetch_inquirer_article(href)
                if article:
                    articles.append(article)
                    log.info(f"  [{len(articles):>3}/{limit}] {article.title[:70]}")
                    sleep()
                if len(articles) >= limit:
                    break
            if found == 0:
                break
            sleep()

    log.info(f"  Collected {len(articles)} articles from Inquirer\n")
    return articles


def _fetch_inquirer_article(url: str) -> Article | None:
    soup = get(url)
    if not soup:
        return None

    title_tag = soup.select_one("h1.entry-title, h1#article-headline, h1")
    title = clean(title_tag.get_text()) if title_tag else ""
    if not title or len(title) < 10:
        return None

    text = body_text(soup, [
        "div#article-content",
        "div.entry-content",
        "div#story-body",
        "article",
    ])
    if len(text) < 150:
        return None

    date_tag = soup.select_one("time, .date-published, .entry-date")
    date = date_tag.get("datetime", date_tag.get_text()) if date_tag else ""

    return Article(title=title, text=text, label="real",
                   source="Inquirer", url=url, date=clean(date))


# ---------------------------------------------------------------------------
# GMA News — real-labeled  (fixed: direct category pages)
# ---------------------------------------------------------------------------

def scrape_gma(limit: int) -> list[Article]:
    """GMA News — category pages use JS; only the main /news/ listing has story links."""
    log.info("=== GMA News ===")
    articles: list[Article] = []
    seen: set[str] = set()

    base = "https://www.gmanetwork.com/news/"

    for page in range(1, 40):
        if len(articles) >= limit:
            break
        url = base if page == 1 else f"{base}?page={page}"
        log.info(f"  Page {page}: {url}")
        soup = get(url)
        if not soup:
            break

        links = [
            a.get("href", "")
            for a in soup.find_all("a", href=True)
            if "gmanetwork.com/news/" in a.get("href", "")
            and "/story/" in a.get("href", "")
        ]
        found = 0
        for href in links:
            if href in seen:
                continue
            seen.add(href)
            found += 1
            article = _fetch_gma_article(href)
            if article:
                articles.append(article)
                log.info(f"  [{len(articles):>3}/{limit}] {article.title[:70]}")
                sleep()
            if len(articles) >= limit:
                break
        if found == 0:
            break
        sleep()

    log.info(f"  Collected {len(articles)} articles from GMA News\n")
    return articles


def _fetch_gma_article(url: str) -> Article | None:
    soup = get(url)
    if not soup:
        return None

    title_tag = soup.select_one(
        "h1.story-title, h1.article-title, h1.title, h1"
    )
    title = clean(title_tag.get_text()) if title_tag else ""
    if not title or len(title) < 10:
        return None

    text = body_text(soup, [
        "div.story-text",
        "div.story_content",
        "div#story_content",
        "div.article-content",
        "article",
    ])
    if len(text) < 150:
        return None

    date_tag = soup.select_one("time, .dateline, .story_date, .story-date")
    date = date_tag.get("datetime", date_tag.get_text()) if date_tag else ""

    return Article(title=title, text=text, label="real",
                   source="GMA News", url=url, date=clean(date))


# ---------------------------------------------------------------------------
# Manila Bulletin — real-labeled
# ---------------------------------------------------------------------------

MB_SECTIONS = [
    "https://mb.com.ph/category/news/",
    "https://mb.com.ph/category/business/",
    "https://mb.com.ph/category/technology/",
    "https://mb.com.ph/category/nation/",
    "https://mb.com.ph/category/world/",
    "https://mb.com.ph/category/entertainment/",
    "https://mb.com.ph/category/sports/",
]


def scrape_manila_bulletin(limit: int) -> list[Article]:
    log.info("=== Manila Bulletin ===")
    articles: list[Article] = []
    seen: set[str] = set()

    for section_url in MB_SECTIONS:
        if len(articles) >= limit:
            break
        for page in range(1, 20):
            if len(articles) >= limit:
                break
            paged = section_url if page == 1 else f"{section_url}page/{page}/"
            log.info(f"  Page {page}: {paged}")
            soup = get(paged)
            if not soup:
                break
            links = soup.select("h2 a, h3 a, .article-title a, .entry-title a, .post-title a, h4 a")
            found = 0
            for a in links:
                href = a.get("href", "")
                if not href or href in seen or "mb.com.ph" not in href:
                    continue
                seen.add(href)
                found += 1
                article = _fetch_manila_bulletin_article(href)
                if article:
                    articles.append(article)
                    log.info(f"  [{len(articles):>3}/{limit}] {article.title[:70]}")
                    sleep()
                if len(articles) >= limit:
                    break
            if found == 0:
                break
            sleep()

    log.info(f"  Collected {len(articles)} articles from Manila Bulletin\n")
    return articles


def _fetch_manila_bulletin_article(url: str) -> Article | None:
    soup = get(url)
    if not soup:
        return None

    title_tag = soup.select_one("h1.entry-title, h1.post-title, h1.article-title, h1")
    title = clean(title_tag.get_text()) if title_tag else ""
    if not title or len(title) < 10:
        return None

    text = body_text(soup, [
        "div.entry-content",
        "div.article-content",
        "div.post-content",
        "div.td-post-content",
        "article",
    ])
    if len(text) < 150:
        return None

    date_tag = soup.select_one("time, .entry-date, .post-date, .published")
    date = date_tag.get("datetime", date_tag.get_text()) if date_tag else ""

    return Article(title=title, text=text, label="real",
                   source="Manila Bulletin", url=url, date=clean(date))


# ---------------------------------------------------------------------------
# Philstar — real-labeled
# ---------------------------------------------------------------------------

PHILSTAR_SECTIONS = [
    "https://www.philstar.com/headlines",
    "https://www.philstar.com/nation",
    "https://www.philstar.com/business",
    "https://www.philstar.com/technology",
    "https://www.philstar.com/world",
    "https://www.philstar.com/opinion",
    "https://www.philstar.com/entertainment",
    "https://www.philstar.com/sports",
]


def scrape_philstar(limit: int) -> list[Article]:
    log.info("=== Philstar ===")
    articles: list[Article] = []
    seen: set[str] = set()

    for section_url in PHILSTAR_SECTIONS:
        if len(articles) >= limit:
            break
        for page in range(1, 20):
            if len(articles) >= limit:
                break
            paged = section_url if page == 1 else f"{section_url}?page={page}"
            log.info(f"  Page {page}: {paged}")
            soup = get(paged)
            if not soup:
                break
            links = soup.select("h2 a, h3 a, .news-title a, .article-title a, .headline a, h4 a")
            found = 0
            for a in links:
                href = a.get("href", "")
                if not href or href in seen:
                    continue
                if not href.startswith("http"):
                    href = "https://www.philstar.com" + href
                if "philstar.com" not in href:
                    continue
                seen.add(href)
                found += 1
                article = _fetch_philstar_article(href)
                if article:
                    articles.append(article)
                    log.info(f"  [{len(articles):>3}/{limit}] {article.title[:70]}")
                    sleep()
                if len(articles) >= limit:
                    break
            if found == 0:
                break
            sleep()

    log.info(f"  Collected {len(articles)} articles from Philstar\n")
    return articles


def _fetch_philstar_article(url: str) -> Article | None:
    soup = get(url)
    if not soup:
        return None

    title_tag = soup.select_one("h1.article__title, h1.title, h1")
    title = clean(title_tag.get_text()) if title_tag else ""
    if not title or len(title) < 10:
        return None

    text = body_text(soup, [
        "div.article__body-text",
        "div#sports_article_body",
        "div.article-content",
        "div.entry-content",
        "article",
    ])
    if len(text) < 150:
        return None

    date_tag = soup.select_one("time, .article__date, .date-published, .timestamp")
    date = date_tag.get("datetime", date_tag.get_text()) if date_tag else ""

    return Article(title=title, text=text, label="real",
                   source="Philstar", url=url, date=clean(date))


# ---------------------------------------------------------------------------
# TSEK.PH — fake-labeled  (WordPress-based fact-check site)
# ---------------------------------------------------------------------------

TSEK_ARCHIVE_ROOTS = [
    "https://tsek.ph/",
    "https://tsek.ph/category/fact-check/",
    "https://tsek.ph/category/debunked/",
    "https://tsek.ph/category/misleading/",
    "https://tsek.ph/category/missing-context/",
    "https://tsek.ph/category/false/",
    "https://tsek.ph/category/satire/",
]

_TSEK_SKIP = {
    "tsek.ph/page/", "tsek.ph/category/", "tsek.ph/tag/",
    "tsek.ph/author/", "tsek.ph/#", "tsek.ph/about",
    "tsek.ph/contact", "tsek.ph/privacy", "tsek.ph/wp-",
}


def _is_tsek_article_url(href: str) -> bool:
    if not href or "tsek.ph" not in href:
        return False
    if any(skip in href for skip in _TSEK_SKIP):
        return False
    if href.rstrip("/") in {"https://tsek.ph", "http://tsek.ph"}:
        return False
    # Article URLs have at least one path segment beyond the domain
    parts = href.split("tsek.ph/", 1)
    if len(parts) < 2 or not parts[1].strip("/"):
        return False
    return True


def scrape_tsek(limit: int) -> list[Article]:
    log.info("=== TSEK.PH Fact Check ===")
    articles: list[Article] = []
    seen: set[str] = set()

    for root in TSEK_ARCHIVE_ROOTS:
        if len(articles) >= limit:
            break
        for page in range(1, 80):
            if len(articles) >= limit:
                break
            if page == 1:
                url = root
            else:
                # WordPress pagination: /category/fact-check/page/2/
                base = root.rstrip("/")
                url = f"{base}/page/{page}/"
            log.info(f"  Listing page {page}  ({url})")
            soup = get(url)
            if not soup:
                break

            links = [
                a.get("href", "")
                for a in soup.find_all("a", href=True)
                if _is_tsek_article_url(a.get("href", ""))
            ]
            if not links:
                log.info("  No article links found — stopping")
                break

            new_links = 0
            for href in links:
                if href in seen:
                    continue
                seen.add(href)
                new_links += 1
                article = _fetch_tsek_article(href)
                if article:
                    articles.append(article)
                    log.info(f"  [{len(articles):>3}/{limit}] {article.title[:70]}")
                    sleep()
                if len(articles) >= limit:
                    break

            if new_links == 0:
                log.info("  No new links on this root — next")
                break

            sleep()

    log.info(f"  Collected {len(articles)} articles from TSEK.PH\n")
    return articles


def _fetch_tsek_article(url: str) -> Article | None:
    soup = get(url)
    if not soup:
        return None

    title_tag = soup.select_one("h1.entry-title, h1.post-title, h1.article-title, h1")
    title = clean(title_tag.get_text()) if title_tag else ""
    if not title or len(title) < 10:
        return None

    text = body_text(soup, [
        "div.entry-content",
        "div.post-content",
        "div.article-content",
        "article",
    ])
    if len(text) < 150:
        return None

    date_tag = soup.select_one("time, .entry-date, .post-date, .published")
    date = date_tag.get("datetime", date_tag.get_text()) if date_tag else ""

    return Article(title=title, text=text, label="fake",
                   source="TSEK.PH", url=url, date=clean(date))


# ---------------------------------------------------------------------------
# AFP Fact Check Philippines — fake-labeled
# ---------------------------------------------------------------------------

AFP_LISTING_PAGES = [
    "https://factcheck.afp.com/list/all/all/PHL",
    "https://factcheck.afp.com/list/all/all/PHL?page=2",
    "https://factcheck.afp.com/list/all/all/PHL?page=3",
    "https://factcheck.afp.com/list/all/all/PHL?page=4",
    "https://factcheck.afp.com/list/all/all/PHL?page=5",
]


def _is_afp_article_url(href: str) -> bool:
    if not href or "factcheck.afp.com" not in href:
        return False
    skip = ["/list/", "/tag/", "/author/", "/search", "/#", "/about", "/contact"]
    if any(s in href for s in skip):
        return False
    parts = href.split("factcheck.afp.com/", 1)
    if len(parts) < 2 or not parts[1].strip("/"):
        return False
    return True


def scrape_afp(limit: int) -> list[Article]:
    log.info("=== AFP Fact Check PH ===")
    articles: list[Article] = []
    seen: set[str] = set()

    for listing_url in AFP_LISTING_PAGES:
        if len(articles) >= limit:
            break
        log.info(f"  Listing: {listing_url}")
        soup = get(listing_url)
        if not soup:
            continue

        links = [
            a.get("href", "")
            for a in soup.find_all("a", href=True)
            if _is_afp_article_url(a.get("href", ""))
        ]
        if not links:
            log.info("  No article links found — stopping")
            break

        for href in links:
            if len(articles) >= limit:
                break
            if href in seen:
                continue
            seen.add(href)
            article = _fetch_afp_article(href)
            if article:
                articles.append(article)
                log.info(f"  [{len(articles):>3}/{limit}] {article.title[:70]}")
                sleep()

    log.info(f"  Collected {len(articles)} articles from AFP Fact Check PH\n")
    return articles


def _fetch_afp_article(url: str) -> Article | None:
    soup = get(url)
    if not soup:
        return None

    title_tag = soup.select_one("h1.article-title, h1.title, h1.entry-title, h1")
    title = clean(title_tag.get_text()) if title_tag else ""
    if not title or len(title) < 10:
        return None

    text = body_text(soup, [
        "div.article-content",
        "div.article__body",
        "div.entry-content",
        "article",
    ])
    if len(text) < 150:
        return None

    date_tag = soup.select_one("time, .article-date, .publish-date, .date")
    date = date_tag.get("datetime", date_tag.get_text()) if date_tag else ""

    return Article(title=title, text=text, label="fake",
                   source="AFP Fact Check PH", url=url, date=clean(str(date)))


# ---------------------------------------------------------------------------
# ABS-CBN News — real-labeled
# ---------------------------------------------------------------------------

ABSCBN_RSS_FEEDS = [
    "https://news.abs-cbn.com/rss/news",
    "https://news.abs-cbn.com/rss/nation",
    "https://news.abs-cbn.com/rss/business",
    "https://news.abs-cbn.com/rss/world",
    "https://news.abs-cbn.com/rss/entertainment",
]


def scrape_abscbn(limit: int) -> list[Article]:
    log.info("=== ABS-CBN News (RSS) ===")
    articles: list[Article] = []
    seen: set[str] = set()

    for feed_url in ABSCBN_RSS_FEEDS:
        if len(articles) >= limit:
            break
        log.info(f"  Feed: {feed_url}")
        items = get_rss(feed_url)
        log.info(f"  Found {len(items)} items in feed")
        for item in items:
            if len(articles) >= limit:
                break
            href = item["link"]
            if not href or href in seen:
                continue
            seen.add(href)
            article = _fetch_abscbn_article(href, item["title"], item["date"])
            if article:
                articles.append(article)
                log.info(f"  [{len(articles):>3}/{limit}] {article.title[:70]}")
                sleep()

    log.info(f"  Collected {len(articles)} articles from ABS-CBN News\n")
    return articles


def _fetch_abscbn_article(url: str, rss_title: str = "", rss_date: str = "") -> Article | None:
    soup = get(url)
    if not soup:
        return None

    title_tag = soup.select_one("h1.article-title, h1.story-title, h1.post-title, h1")
    title = clean(title_tag.get_text()) if title_tag else rss_title
    if not title or len(title) < 10:
        return None

    text = body_text(soup, [
        "div.article-content-body",
        "div.article__body",
        "div.story-body",
        "div.entry-content",
        "article",
    ])
    if len(text) < 150:
        return None

    date_tag = soup.select_one("time, .article-date, .dateline, .publish-date")
    date = date_tag.get("datetime", date_tag.get_text()) if date_tag else rss_date

    return Article(title=title, text=text, label="real",
                   source="ABS-CBN News", url=url, date=clean(str(date)))


# ---------------------------------------------------------------------------
# Philippine News Agency (PNA) — real-labeled
# ---------------------------------------------------------------------------

PNA_RSS_FEEDS = [
    "https://www.pna.gov.ph/rss",
    "https://www.pna.gov.ph/categories/1/articles.rss",
    "https://www.pna.gov.ph/categories/2/articles.rss",
    "https://www.pna.gov.ph/categories/3/articles.rss",
    "https://www.pna.gov.ph/categories/4/articles.rss",
]


def scrape_pna(limit: int) -> list[Article]:
    log.info("=== Philippine News Agency — PNA (RSS) ===")
    articles: list[Article] = []
    seen: set[str] = set()

    for feed_url in PNA_RSS_FEEDS:
        if len(articles) >= limit:
            break
        log.info(f"  Feed: {feed_url}")
        items = get_rss(feed_url)
        log.info(f"  Found {len(items)} items in feed")
        for item in items:
            if len(articles) >= limit:
                break
            href = item["link"]
            if not href or href in seen:
                continue
            seen.add(href)
            article = _fetch_pna_article(href, item["title"], item["date"])
            if article:
                articles.append(article)
                log.info(f"  [{len(articles):>3}/{limit}] {article.title[:70]}")
                sleep()

    log.info(f"  Collected {len(articles)} articles from PNA\n")
    return articles


def _fetch_pna_article(url: str, rss_title: str = "", rss_date: str = "") -> Article | None:
    soup = get(url)
    if not soup:
        return None

    title_tag = soup.select_one("h1.article-title, h1.title, h1")
    title = clean(title_tag.get_text()) if title_tag else rss_title
    if not title or len(title) < 10:
        return None

    text = body_text(soup, [
        "div.article-content",
        "div.field-items",
        "div.body",
        "div.content",
        "article",
    ])
    if len(text) < 150:
        return None

    date_tag = soup.select_one("time, .date, .published, .article-date")
    date = date_tag.get("datetime", date_tag.get_text()) if date_tag else rss_date

    return Article(title=title, text=text, label="real",
                   source="Philippine News Agency", url=url, date=clean(str(date)))


# ---------------------------------------------------------------------------
# Save to CSV (appends to existing, deduplicates by URL)
# ---------------------------------------------------------------------------

def load_existing_urls(path: str) -> set[str]:
    seen: set[str] = set()
    try:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                seen.add(row.get("url", ""))
    except FileNotFoundError:
        pass
    return seen


def save_csv(articles: list[Article], path: str, append: bool = False):
    fieldnames = [f.name for f in fields(Article)]
    mode = "a" if append else "w"
    write_header = not append
    try:
        import os
        if append and not os.path.exists(path):
            write_header = True
    except Exception:
        pass

    with open(path, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for art in articles:
            writer.writerow({
                "title":          art.title,
                "text":           art.text,
                "label":          art.label,
                "source":         art.source,
                "url":            art.url,
                "date":           art.date,
                "source_dataset": art.source_dataset,
            })
    log.info(f"Saved {len(articles)} articles → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Philippine Fake News Scraper")
    parser.add_argument("--fake", type=int, default=DEFAULT_FAKE_PER_SOURCE,
                        help="Target articles per fake source (VERA Files, Rappler)")
    parser.add_argument("--real", type=int, default=DEFAULT_REAL_PER_SOURCE,
                        help="Target articles per real source (Inquirer, GMA, MB, Philstar)")
    parser.add_argument("--append", action="store_true",
                        help="Append to existing dataset.csv instead of overwriting")
    args = parser.parse_args()

    fake_limit = args.fake
    real_limit = args.real

    log.info("Starting Philippine Fake News Scraper v4")
    log.info(f"  Target fake per source : {fake_limit}  (VERA Files + Rappler + TSEK.PH + AFP = up to {fake_limit * 4})")
    log.info(f"  Target real per source : {real_limit}  (6 sources = up to {real_limit * 6})")
    log.info(f"  Grand total target     : ~{fake_limit * 4 + real_limit * 6}")
    log.info(f"  Mode                   : {'append' if args.append else 'overwrite'}\n")

    # Always load existing URLs for deduplication
    existing_urls: set[str] = load_existing_urls(OUTPUT_FILE)
    if existing_urls:
        log.info(f"  Existing articles in CSV: {len(existing_urls)}\n")

    total_saved = 0

    def scrape_and_save(articles: list[Article], source_name: str):
        nonlocal total_saved
        new = [a for a in articles if a.url not in existing_urls]
        for a in new:
            existing_urls.add(a.url)
        if new:
            save_csv(new, OUTPUT_FILE, append=True)
            total_saved += len(new)
            log.info(f"  --> Saved {len(new)} new articles from {source_name}  (total in CSV: {len(existing_urls)})\n")
        else:
            log.info(f"  --> No new articles from {source_name} (all duplicates)\n")

    scrape_and_save(scrape_vera_files(fake_limit),      "VERA Files")
    scrape_and_save(scrape_rappler(fake_limit),         "Rappler")
    scrape_and_save(scrape_tsek(fake_limit),            "TSEK.PH")
    scrape_and_save(scrape_afp(fake_limit),             "AFP Fact Check PH")

    scrape_and_save(scrape_inquirer(real_limit),        "Inquirer")
    scrape_and_save(scrape_gma(real_limit),             "GMA News")
    scrape_and_save(scrape_manila_bulletin(real_limit), "Manila Bulletin")
    scrape_and_save(scrape_philstar(real_limit),        "Philstar")
    scrape_and_save(scrape_abscbn(real_limit),          "ABS-CBN News")
    scrape_and_save(scrape_pna(real_limit),             "PNA")

    log.info("\n=== Final Summary ===")
    log.info(f"  New articles saved this run : {total_saved}")
    log.info(f"  Total articles in CSV       : {len(existing_urls)}")
    log.info("\nDone! Run: bash retrain.sh")


if __name__ == "__main__":
    main()
