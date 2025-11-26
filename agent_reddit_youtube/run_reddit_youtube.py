import os
import json
import requests
from datetime import datetime, timezone
from collections import Counter
from typing import List, Dict, Any

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# ========= CONFIG =========
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

# Simple knobs to control how much data we fetch
REDDIT_LIMIT = 10          # number of Reddit posts/comments per term
YOUTUBE_VIDEOS = 10        # number of YouTube videos per term
YOUTUBE_COMMENTS = 10     # number of comments per video
BATCH_SIZE = 8            # number of items per LLM call


# ========= HELPERS =========

def detect_entities(text: str) -> List[str]:
    """
    Simple rule-based entity detection:
    returns a list containing 'Taboola' and/or 'Realize' if they appear in the text.
    """
    t = text.lower()
    entities = []
    if "taboola" in t:
        entities.append("Taboola")
    if "realize" in t:
        entities.append("Realize")
    return entities


# ========= REDDIT FETCH =========

def fetch_reddit(term: str, limit: int) -> List[Dict[str, Any]]:
    """
    Fetch recent Reddit posts and comments that match the given search term.
    Returns a list of normalized items with:
    id, platform, source, text, url, created_at (ISO string).
    """
    headers = {"User-Agent": "Mozilla/5.0 (social-listening-agent)"}
    items: List[Dict[str, Any]] = []

    # --- Posts search ---
    params = {"q": term, "limit": limit, "sort": "new", "type": "link"}
    r = requests.get("https://www.reddit.com/search.json", params=params, headers=headers)
    if r.status_code == 200:
        for child in r.json().get("data", {}).get("children", []):
            d = child["data"]
            text = (d.get("title") or "") + "\n\n" + (d.get("selftext") or "")
            items.append({
                "id": "reddit_post_" + d["id"],
                "platform": "reddit",
                "source": "post",
                "text": text.strip(),
                "url": "https://www.reddit.com" + d.get("permalink", ""),
                "created_at": datetime.fromtimestamp(
                    d.get("created_utc", 0),
                    timezone.utc
                ).isoformat(),
            })

    # --- Comments search ---
    params["type"] = "comment"
    r = requests.get("https://www.reddit.com/search.json", params=params, headers=headers)
    if r.status_code == 200:
        for child in r.json().get("data", {}).get("children", []):
            d = child["data"]
            items.append({
                "id": "reddit_comment_" + d["id"],
                "platform": "reddit",
                "source": "comment",
                "text": d.get("body", "").strip(),
                "url": "https://www.reddit.com" + d.get("permalink", ""),
                "created_at": datetime.fromtimestamp(
                    d.get("created_utc", 0),
                    timezone.utc
                ).isoformat(),
            })

    return items


# ========= YOUTUBE SEARCH + COMMENTS =========

def youtube_search(term: str, max_results: int) -> List[str]:
    """
    Search YouTube for videos that match the given term.
    Returns a list of video IDs.
    """
    if not YOUTUBE_API_KEY:
        return []

    url = "https://www.googleapis.com/youtube/v3/search"
    params = {
        "q": term,
        "part": "snippet",
        "type": "video",
        "maxResults": max_results,
        "key": YOUTUBE_API_KEY
    }
    r = requests.get(url, params=params)
    if r.status_code != 200:
        return []

    vids: List[str] = []
    for item in r.json().get("items", []):
        vid = item["id"].get("videoId")
        if vid:
            vids.append(vid)
    return vids


def youtube_comments(video_id: str, max_comments: int) -> List[Dict[str, Any]]:
    """
    Fetch YouTube top-level comments for a given video.
    Returns normalized items with id, platform, source, text, url, created_at.
    """
    if not YOUTUBE_API_KEY:
        return []

    url = "https://www.googleapis.com/youtube/v3/commentThreads"
    params = {
        "videoId": video_id,
        "part": "snippet",
        "maxResults": max_comments,
        "textFormat": "plainText",
        "key": YOUTUBE_API_KEY
    }
    r = requests.get(url, params=params)
    if r.status_code != 200:
        return []

    out: List[Dict[str, Any]] = []
    for item in r.json().get("items", []):
        snip = item["snippet"]["topLevelComment"]["snippet"]
        out.append({
            "id": "yt_comment_" + item["id"],
            "platform": "youtube",
            "source": "comment",
            "text": snip.get("textDisplay", "").strip(),
            "url": f"https://www.youtube.com/watch?v={video_id}",
            "created_at": snip.get("publishedAt", datetime.now(timezone.utc).isoformat())
        })
    return out


# ========= BATCH LLM =========

def get_llm_chain():
    """
    Build a LangChain pipeline:
    PromptTemplate -> ChatOpenAI (with JSON schema response_format).
    The model must return a JSON object that matches 'result_schema'.
    """
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY missing")

    # Explicit JSON schema for the LLM response.
    # This is used with OpenAI response_format = "json_schema" to reduce parsing errors.
    result_schema = {
        "name": "SentimentResults",
        "schema": {
            "type": "object",
            "properties": {
                "results": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "entities": {
                                "type": "array",
                                "items": {"type": "string"},
                                "minItems": 1
                            },
                            "overall_sentiment": {
                                "type": "string",
                                "enum": ["positive", "neutral", "negative"]
                            },
                            "fields": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "field": {
                                            "type": "string",
                                            "enum": [
                                                "performance",
                                                "UX",
                                                "support",
                                                "ads quality",
                                                "pricing",
                                                "integration",
                                                "other"
                                            ]
                                        },
                                        "sentiment": {
                                            "type": "string",
                                            "enum": ["positive", "neutral", "negative"]
                                        },
                                        "confidence": {
                                            "type": "integer",
                                            "minimum": 0,
                                            "maximum": 100
                                        }
                                    },
                                    "required": ["field", "sentiment", "confidence"],
                                    "additionalProperties": False
                                },
                                "minItems": 1,
                                "maxItems": 3
                            },
                            "summary": {"type": "string"},
                            "topics": {
                                "type": "array",
                                "items": {"type": "string"},
                                "minItems": 1,
                                "maxItems": 4
                            }
                        },
                        "required": [
                            "id",
                            "entities",
                            "overall_sentiment",
                            "fields",
                            "summary",
                            "topics"
                        ],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["results"],
            "additionalProperties": False
        },
        "strict": True
    }

    # Short, focused system prompt.
    system_prompt = """
You are analyzing online content about Taboola and Realize.

Input:
- One JSON array called "items".
- Each item has: "id", "text", "platform".

Task:
- For EVERY input item, return exactly one result in "results".
- Fill: entities, overall_sentiment, fields (1-3), summary, topics (1-4).
- If only Taboola is mentioned → entities = ["Taboola"].
- If only Realize → ["Realize"].
- If both → ["Taboola", "Realize"].

Return JSON only, matching the JSON schema.
""".strip()

    human_prompt = """
Analyze the following JSON array under the key "items":

{items_json}
""".strip()

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_prompt)
    ])

    llm = ChatOpenAI(
        model=OPENAI_MODEL,
        temperature=0,
        api_key=OPENAI_API_KEY,
        model_kwargs={
            "response_format": {
                "type": "json_schema",
                "json_schema": result_schema
            }
        },
    )

    # Return a chain that takes {"items_json": "..."} and returns a model response.
    return prompt | llm


def safe_json_load(s: str):
    """
    Defensive JSON loader.
    If full string fails, tries to load the substring from the first '{' to the last '}'.
    (Should be mostly unnecessary with response_format=json_schema, but kept as extra safety.)
    """
    s = s.strip()
    try:
        return json.loads(s)
    except Exception:
        first = s.find("{")
        last = s.rfind("}")
        if first != -1 and last != -1:
            return json.loads(s[first:last + 1])
        raise


def analyze_batched(items: List[Dict[str, Any]], batch: int = 8) -> List[Dict[str, Any]]:
    """
    Run the LLM in batches over the collected items.
    For each item, attach:
      - entities
      - overall_sentiment
      - fields[]
      - summary
      - topics[]
    Fallback: if the LLM returns nothing for an item, use keyword-based entity detection
    and set neutral sentiment with empty fields/topics.
    """
    chain = get_llm_chain()
    results: List[Dict[str, Any]] = []

    for i in range(0, len(items), batch):
        sub = items[i:i + batch]

        # Minimal payload sent to the LLM.
        payload = [
            {
                "id": x["id"],
                "text": x["text"],
                "platform": x["platform"]
            }
            for x in sub
        ]

        parsed: List[Dict[str, Any]] = []
        try:
            msg = chain.invoke({"items_json": json.dumps(payload, ensure_ascii=False)})
            data = safe_json_load(msg.content)
            parsed = data.get("results", [])
        except Exception as e:
            print(f"LLM batch error: {e}")
            parsed = []

        # Build a map from item id -> LLM output
        out_map: Dict[str, Dict[str, Any]] = {}
        for p in parsed:
            if not isinstance(p, dict):
                continue
            pid = p.get("id")
            if not pid:
                continue
            out_map[pid] = p

        # Merge original items with LLM output (or fallback)
        for original in sub:
            rid = original["id"]
            r = out_map.get(rid)

            # Fallback if we have no LLM result for this item
            if r is None:
                ents = detect_entities(original.get("text", ""))
                if not ents:
                    # If the text does not mention Taboola/Realize at all, skip it.
                    continue
                r = {
                    "id": rid,
                    "entities": ents,
                    "overall_sentiment": "neutral",
                    "fields": [],
                    "summary": "",
                    "topics": []
                }

            ents = r.get("entities") or []
            if not ents:
                # If LLM output has no entities, skip (not relevant to dashboard).
                continue

            results.append({
                **original,
                "entities": ents,
                "overall_sentiment": r.get("overall_sentiment", "neutral"),
                "fields": r.get("fields") or [],
                "summary": r.get("summary") or "",
                "topics": r.get("topics") or [],
            })

    # Only keep items that actually have entities
    return [r for r in results if r.get("entities")]


# ========= AGGREGATIONS =========

def agg_sentiment_per_field(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, int]]]:
    """
    Aggregate sentiment counts per entity and field.
    Output structure:
      {
        "Taboola": {
          "performance": {"positive": 3, "neutral": 1, "negative": 2},
          ...
        },
        "Realize": { ... }
      }
    """
    out: Dict[str, Dict[str, Dict[str, int]]] = {}
    for rec in records:
        for ent in rec["entities"]:
            out.setdefault(ent, {})
            for f in rec["fields"]:
                field = f["field"]
                sent = f["sentiment"]
                ent_map = out[ent].setdefault(field, {"positive": 0, "neutral": 0, "negative": 0})
                if sent in ent_map:
                    ent_map[sent] += 1
    return out


def agg_trend(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, int]]]:
    """
    Build a sentiment trend over time (by *week*) for each entity.
    Output structure:
      {
        "Taboola": {
          "2025-W03": {"positive": 3, "neutral": 0, "negative": 1},
          ...
        },
        "Realize": { ... }
      }
    """
    out: Dict[str, Dict[str, Dict[str, int]]] = {}

    for rec in records:
        created_at = rec.get("created_at")
        if not created_at:
            continue

        # Parse to datetime object
        try:
            dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        except Exception:
            continue

        # Get ISO week
        year, week, _ = dt.isocalendar()
        week_key = f"{year}-W{week:02d}"

        for ent in rec.get("entities", []):
            out.setdefault(ent, {})
            bucket = out[ent].setdefault(
                week_key,
                {"positive": 0, "neutral": 0, "negative": 0}
            )

            sent = rec.get("overall_sentiment", "neutral")
            if sent in bucket:
                bucket[sent] += 1

    return out


def agg_top_themes(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute top 3 topics ('themes') per entity based on the 'topics' array.
    Output structure:
      {
        "Taboola": [{"theme": "pricing", "count": 4}, ... up to 3],
        "Realize": [...]
      }
    """
    out: Dict[str, Counter] = {}
    for rec in records:
        for ent in rec["entities"]:
            out.setdefault(ent, Counter())
            out[ent].update([t.lower() for t in rec.get("topics", [])])

    final: Dict[str, Any] = {}
    for ent, counter in out.items():
        top = counter.most_common(3)
        final[ent] = [{"theme": t, "count": c} for t, c in top]
    return final


# ========= PIPELINE =========

def run_pipeline() -> None:
    """
    Full end-to-end pipeline:
    1. Fetch Reddit + YouTube content for "Taboola" and "Realize".
    2. Deduplicate items.
    3. Call LLM in batches to attach structured sentiment & topics.
    4. Compute aggregations.
    5. Save everything into results.json for the HTML dashboard.
    """
    all_items: List[Dict[str, Any]] = []

    # 1) Reddit
    for term in ["Taboola", "Realize"]:
        all_items.extend(fetch_reddit(term, REDDIT_LIMIT))

    # 2) YouTube (bonus source)
    for term in ["Taboola", "Realize"]:
        vids = youtube_search(term, YOUTUBE_VIDEOS)
        for vid in vids:
            all_items.extend(youtube_comments(vid, YOUTUBE_COMMENTS))

    # 3) Deduplicate by id (simple dict overwrite)
    dedup: Dict[str, Dict[str, Any]] = {}
    for x in all_items:
        dedup[x["id"]] = x
    items = list(dedup.values())

    print(f"Collected {len(items)} items")

    # 4) LLM batch sentiment analysis
    enriched = analyze_batched(items, BATCH_SIZE)
    print(f"Processed {len(enriched)} items via LLM")

    # 5) Aggregations for dashboard
    agg1 = agg_sentiment_per_field(enriched)
    agg2 = agg_trend(enriched)
    agg3 = agg_top_themes(enriched)

    final = {
        "items": enriched,
        "sentiment_distribution": agg1,
        "trend_by_date": agg2,
        "top_themes": agg3
    }

    # 6) Save JSON output
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)

    print("Saved results.json")


# ========= MAIN =========

if __name__ == "__main__":
    run_pipeline()

