import os
import json
import datetime
from typing import List, Dict, Any
from collections import Counter, defaultdict

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# -------------------------------------------------------------------
# Config & environment
# -------------------------------------------------------------------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

if not OPENAI_API_KEY:
    raise ValueError("Missing OPENAI_API_KEY. Set it in your environment or .env file.")

TARGET_ENTITIES = ["Taboola", "Realize"]
FIELDS = ["product", "performance", "business", "brand"]

# JSON schema used with OpenAI response_format to enforce structure
SENTIMENT_SCHEMA = {
    "name": "SentimentAnalysis",
    "schema": {
        "type": "object",
        "properties": {
            "fields": {
                "type": "object",
                "properties": {
                    "product": {"type": "string", "enum": ["positive", "neutral", "negative"]},
                    "performance": {"type": "string", "enum": ["positive", "neutral", "negative"]},
                    "business": {"type": "string", "enum": ["positive", "neutral", "negative"]},
                    "brand": {"type": "string", "enum": ["positive", "neutral", "negative"]},
                },
                "required": ["product", "performance", "business", "brand"],
            },
            "overall_sentiment": {
                "type": "string",
                "enum": ["positive", "neutral", "negative"],
            },
            "themes": {
                "type": "array",
                "items": {"type": "string"},
            },
            "quote": {
                "type": "string",
            },
        },
        "required": ["fields", "overall_sentiment", "themes", "quote"],
        "additionalProperties": False,
    },
}


# -------------------------------------------------------------------
# Data ingestion: Reddit
# -------------------------------------------------------------------

def fetch_reddit_mentions(keywords: List[str], limit: int = 30) -> List[Dict]:
    """
    Fetch recent Reddit posts containing the given keywords using the public search API.
    Returns a list of normalized records.
    """
    import requests
    import time

    out = []
    base_url = "https://www.reddit.com/search.json"
    headers = {
        "User-Agent": "social-listening-agent/0.1 by your_username"
    }

    for kw in keywords:
        params = {
            "q": kw,
            "limit": limit,
            "sort": "new",
        }
        try:
            resp = requests.get(base_url, params=params, headers=headers, timeout=10)
            if resp.status_code != 200:
                print("Reddit status code:", resp.status_code, resp.text[:200])
                continue
            data = resp.json()
            for child in data.get("data", {}).get("children", []):
                post = child.get("data", {})
                out.append({
                    "platform": "Reddit",
                    "id": post.get("id"),
                    "entity": kw,
                    "title": post.get("title", "") or "",
                    "body": post.get("selftext", "") or "",
                    "author": post.get("author", "") or "",
                    "created_utc": post.get("created_utc"),
                    "permalink": "https://www.reddit.com" + post.get("permalink", f"/comments/{post.get('id')}/"),
                })
        except Exception as ex:
            print("Reddit fetch error:", ex)
        # Simple rate limiting to be nice to Reddit's API
        time.sleep(1)
    return out


# -------------------------------------------------------------------
# Data ingestion: Hacker News
# -------------------------------------------------------------------

def fetch_hackernews_mentions(keywords: List[str], limit: int = 30) -> List[Dict]:
    """
    Fetch recent Hacker News stories for the given keywords using the Algolia search API.
    Returns a list of normalized records.
    """
    import requests
    import time

    out = []
    url = "https://hn.algolia.com/api/v1/search_by_date"
    for kw in keywords:
        params = {
            "query": kw,
            "tags": "story",
            "hitsPerPage": limit,
        }
        try:
            resp = requests.get(url, params=params, timeout=8)
            if resp.status_code != 200:
                continue
            data = resp.json()
            for hit in data.get("hits", []):
                # Safely parse HN timestamp to Unix epoch
                created_ts = None
                if hit.get("created_at"):
                    try:
                        created_dt = datetime.datetime.strptime(
                            hit["created_at"][:19],
                            "%Y-%m-%dT%H:%M:%S",
                        )
                        created_ts = int(created_dt.timestamp())
                    except Exception as e:
                        print("HN date parse error:", e)
                        created_ts = None

                out.append({
                    "platform": "HackerNews",
                    "id": hit.get("objectID"),
                    "entity": kw,
                    "title": hit.get("title", "") or "",
                    "body": hit.get("story_text", "") or "",
                    "author": hit.get("author", "") or "",
                    "created_utc": created_ts,
                    "permalink": hit.get("url") or f"https://news.ycombinator.com/item?id={hit.get('objectID')}",
                })

        except Exception as ex:
            print("Hacker News error:", ex)
        time.sleep(0.1)
    return out


# -------------------------------------------------------------------
# LLM prompt + analysis
# -------------------------------------------------------------------

def build_single_post_prompt(post: Dict) -> str:
    """
    Short prompt describing the required sentiment labels and fields.
    The actual output format is enforced by the JSON schema via response_format.
    """
    return (
        "Analyze the sentiment of this post about the given entity.\n"
        "- For each field (product, performance, business, brand) choose one: positive, neutral, or negative.\n"
        "- Also provide overall_sentiment, 1–3 themes, and a short representative quote.\n"
        "Use the JSON schema provided via response_format.\n\n"
        f"Entity: {post['entity']}\n"
        f"Title: {post.get('title', '')}\n"
        f"Text: {post.get('body', '')}"
    )


def analyze_with_langchain(posts: List[Dict]) -> List[Dict]:
    """
    Run sentiment analysis with gpt-4o-mini using response_format=json_schema
    so the model returns valid JSON matching SENTIMENT_SCHEMA.
    """
    llm = ChatOpenAI(
        model=OPENAI_MODEL,
        openai_api_key=OPENAI_API_KEY,
        temperature=0.0,
        model_kwargs={
            "response_format": {
                "type": "json_schema",
                "json_schema": SENTIMENT_SCHEMA,
            }
        },
    )

    import time

    results = []
    for post in posts:
        prompt = build_single_post_prompt(post)
        try:
            output = llm.invoke([HumanMessage(content=prompt)])
            # With response_format=json_schema, content should be valid JSON
            res = json.loads(output.content)

            # Defensive defaults if something is missing
            if "fields" not in res:
                res["fields"] = {}
            for f in FIELDS:
                if f not in res["fields"]:
                    res["fields"][f] = "neutral"

            if "overall_sentiment" not in res:
                res["overall_sentiment"] = "neutral"
            if "themes" not in res:
                res["themes"] = []
            if "quote" not in res:
                res["quote"] = ""

            results.append(res)

        except Exception as ex:
            print("LangChain error:", ex)
            results.append({
                "fields": {f: "neutral" for f in FIELDS},
                "overall_sentiment": "neutral",
                "themes": [],
                "quote": "",
            })
        time.sleep(0.7)
    return results


# -------------------------------------------------------------------
# Aggregations: sentiment, themes, trend
# -------------------------------------------------------------------

def aggregate_results(all_records: List[Dict]) -> Dict:
    """
    Compute aggregated metrics:
    - sentiment distribution per entity × field
    - top 3 themes per entity (with quotes)
    - volume trend over time (by week)
    """
    agg = {
        "sentiment_distribution": {},
        "top_themes": {},
        "trend_over_time": {},
    }

    # Sentiment distribution per entity × field
    for entity in TARGET_ENTITIES:
        agg["sentiment_distribution"][entity] = {}
        for field in FIELDS:
            sentiments = [
                r["analysis"]["fields"].get(field, "neutral")
                for r in all_records
                if r["entity"] == entity and "analysis" in r
            ]
            c = Counter(sentiments)
            tot = sum(c.values()) or 1
            agg["sentiment_distribution"][entity][field] = {
                k: round(100 * v / tot) for k, v in c.items()
            }

    # Top themes and representative quotes per theme
    for entity in TARGET_ENTITIES:
        theme_stats: Dict[str, Dict[str, Any]] = {}  # theme -> {"count": int, "quotes": [..]}
        for r in all_records:
            if r["entity"] != entity or "analysis" not in r:
                continue
            themes = r["analysis"].get("themes", [])
            quote = r["analysis"].get("quote")
            url = r.get("permalink", "")
            platform = r.get("platform")
            for t in themes:
                if t not in theme_stats:
                    theme_stats[t] = {"count": 0, "quotes": []}
                theme_stats[t]["count"] += 1
                if quote:
                    theme_stats[t]["quotes"].append({
                        "text": quote,
                        "url": url,
                        "platform": platform,
                    })

        # Keep top 3 themes by count
        sorted_themes = sorted(
            theme_stats.items(),
            key=lambda kv: kv[1]["count"],
            reverse=True,
        )[:3]

        agg["top_themes"][entity] = [
            {
                "theme": t,
                "count": data["count"],
                "quotes": data["quotes"][:3],  # up to 3 quotes per theme
            }
            for t, data in sorted_themes
        ]

    # Trend over time (by week)
    weekcounts = defaultdict(int)
    for r in all_records:
        ts = r.get("created_utc")
        if ts:
            try:
                dt = datetime.datetime.fromtimestamp(
                    float(ts), tz=datetime.timezone.utc
                )
                # Use ISO week format: YYYY-Www
                year, week, _ = dt.isocalendar()
                week_str = f"{year}-W{week:02d}"
            except Exception:
                week_str = "NA"
            weekcounts[week_str] += 1

    agg["trend_over_time"] = dict(sorted(weekcounts.items()))

    # Include raw records for the frontend table
    agg["all_records"] = all_records
    return agg


# -------------------------------------------------------------------
# Utils: deduplication
# -------------------------------------------------------------------

def deduplicate_records(records: List[Dict]) -> List[Dict]:
    """
    Remove exact duplicates based on (platform, id) so we don't count the same post twice.
    """
    seen = set()
    unique = []
    for r in records:
        key = (r.get("platform"), r.get("id"))
        if key in seen:
            continue
        seen.add(key)
        unique.append(r)
    return unique


# -------------------------------------------------------------------
# Pipeline orchestration
# -------------------------------------------------------------------

def run_pipeline() -> List[Dict]:
    """
    Full pipeline:
    - fetch mentions from Reddit + Hacker News
    - deduplicate
    - run LLM analysis
    - attach analysis back to records
    """
    all_records: List[Dict] = []
    for entity in TARGET_ENTITIES:
        reddit_items = fetch_reddit_mentions([entity], limit=50)
        hn_items = fetch_hackernews_mentions([entity], limit=50)
        all_records += reddit_items + hn_items

    print(f"Fetched {len(all_records)} records (Reddit + Hacker News) before dedup")

    all_records = deduplicate_records(all_records)

    print(f"{len(all_records)} records after dedup")

    posts = [
        {"title": r["title"], "body": r.get("body", ""), "entity": r["entity"]}
        for r in all_records
    ]
    results = analyze_with_langchain(posts)

    for idx, a in enumerate(results):
        all_records[idx]["analysis"] = a

    return all_records


# -------------------------------------------------------------------
# Dashboard generation (HTML from template)
# -------------------------------------------------------------------

def produce_html_dashboard(aggregates: Dict) -> None:
    """
    Generate dashboard.html from a static HTML template (dashboard_template.html),
    replacing simple placeholders with dynamic content.
    """
    records = aggregates.get("all_records", [])
    entities = TARGET_ENTITIES
    platforms = sorted(list(set(r["platform"] for r in records)))
    themes = list(set(
        t
        for ent in entities
        for t in [th["theme"] for th in aggregates["top_themes"][ent]]
    ))

    # 1. Load the HTML template
    with open("dashboard_template.html", "r", encoding="utf-8") as f:
        template = f.read()

    # 2. Build dynamic parts

    # <option> entries for select filters
    entity_options = "".join(
        f"<option value='{e}'>{e}</option>" for e in entities
    )
    platform_options = "".join(
        f"<option value='{p}'>{p}</option>" for p in platforms
    )
    theme_options = "".join(
        f"<option value='{t}'>{t}</option>" for t in themes
    )

    # Sentiment distribution table
    sentiment_html = "<table class='summary-table'><tr><th>Entity</th>"
    for f in FIELDS:
        sentiment_html += f"<th>{f}</th>"
    sentiment_html += "</tr>"
    for ent in entities:
        sentiment_html += f"<tr><td>{ent}</td>"
        for f in FIELDS:
            d = aggregates["sentiment_distribution"][ent].get(f, {})
            colors = {"positive": "#31bd77", "neutral": "#adadad", "negative": "#ea3636"}
            sentiment_html += "<td>"
            if d:
                sentiment_html += " ".join(
                    f"<span style='color:{colors.get(k, '#222')}'>{k}: {v}%</span>"
                    for k, v in d.items()
                )
            sentiment_html += "</td>"
        sentiment_html += "</tr>"
    sentiment_html += "</table>"

    # Top themes + representative quotes per theme
    themes_html = "<div>"
    for ent in entities:
        themes_html += f"<div style='margin-bottom:16px;'><b>{ent}</b>"
        for t in aggregates["top_themes"][ent]:
            theme_name = t["theme"]
            count = t["count"]
            quotes = t.get("quotes", [])
            themes_html += (
                "<div style='padding-left:10px; margin-top:6px; margin-bottom:8px;'>"
                f"<div><span class='theme-badge'>{theme_name}</span> ({count})</div>"
            )
            if quotes:
                themes_html += "<div style='padding-left:15px; margin-top:4px;'>"
                for q in quotes[:3]:
                    plat = q.get("platform", "unknown")
                    url = q.get("url", "#")
                    text = q.get("text", "")
                    themes_html += (
                        "<div class='quote-card'>"
                        f"{text} "
                        f"<span class='platform-badge'>{plat}</span> "
                        f"<a href='{url}' target='_blank' "
                        "style='color:#204080;text-decoration: underline;'>(link)</a>"
                        "</div>"
                    )
                themes_html += "</div>"
            themes_html += "</div>"
        themes_html += "</div>"
    themes_html += "</div>"

    # 3. Replace placeholders in the template
    html = (
        template
        .replace("__ENTITY_OPTIONS__", entity_options)
        .replace("__PLATFORM_OPTIONS__", platform_options)
        .replace("__THEME_OPTIONS__", theme_options)
        .replace("__SENTIMENT_TABLE__", sentiment_html)
        .replace("__THEMES_AND_QUOTES__", themes_html)
        .replace("__POSTS_JSON__", json.dumps(records))
    )

    # 4. Write final HTML file
    with open("dashboard.html", "w", encoding="utf-8") as f:
        f.write(html)

    print("dashboard.html generated from dashboard_template.html!")


# -------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------

def test_analyze_with_langchain() -> None:
    """
    Basic integration test for analyze_with_langchain:
    - Ensures JSON structure matches the schema
    - Ensures fields and overall_sentiment use allowed labels
    """
    if not OPENAI_API_KEY:
        print("Skipping test_analyze_with_langchain: OPENAI_API_KEY not set")
        return

    test_items = [
        {
            "title": "Taboola launches new performance boosting product",
            "body": "The new product is impressive and seems to help advertisers.",
            "entity": "Taboola",
        },
        {
            "title": "Realize faces mixed reviews",
            "body": "Some say the performance is great, others complain about the UX.",
            "entity": "Realize",
        },
    ]

    outputs = analyze_with_langchain(test_items)

    assert isinstance(outputs, list), "Output of analyze_with_langchain should be a list"
    assert len(outputs) == len(test_items), "Number of outputs should match number of inputs"

    allowed_labels = {"positive", "neutral", "negative"}

    for out in outputs:
        assert isinstance(out, dict), "Each result should be a dict"

        # fields
        assert "fields" in out, "Result missing 'fields'"
        assert isinstance(out["fields"], dict), "'fields' must be a dict"

        for f in FIELDS:
            assert f in out["fields"], f"'fields' missing key '{f}'"
            assert out["fields"][f] in allowed_labels, f"Invalid sentiment for field '{f}'"

        # overall_sentiment
        assert "overall_sentiment" in out, "Result missing 'overall_sentiment'"
        assert out["overall_sentiment"] in allowed_labels, "Invalid overall_sentiment value"

        # themes
        assert "themes" in out, "Result missing 'themes'"
        assert isinstance(out["themes"], list), "'themes' must be a list"

        # quote
        assert "quote" in out, "Result missing 'quote'"
        assert isinstance(out["quote"], str), "'quote' must be a string"

    print("✔️ Test_analyze_with_langchain PASSED")


# -------------------------------------------------------------------
# Main entry point
# -------------------------------------------------------------------

if __name__ == "__main__":
    data = run_pipeline()
    with open("all_results.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    aggregates = aggregate_results(data)
    with open("aggregates.json", "w", encoding="utf-8") as f:
        json.dump(aggregates, f, indent=2)

    produce_html_dashboard(aggregates)
    test_analyze_with_langchain()
