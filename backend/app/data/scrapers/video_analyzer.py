"""
Video analysis via YouTube Data API v3 + yt-dlp transcript extraction.
Uses Claude to summarize tactical insights and player form from highlights.
"""
import asyncio
import os
import subprocess
import json
from pathlib import Path
from typing import Optional

import httpx
import anthropic

YOUTUBE_API_BASE = "https://www.googleapis.com/youtube/v3"
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")

_anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", ""))


async def search_team_highlights(team_name: str, max_results: int = 5) -> list[dict]:
    """Search YouTube for recent team highlight videos."""
    if not YOUTUBE_API_KEY:
        print("[video] YOUTUBE_API_KEY not set, skipping video search")
        return []

    query = f"{team_name} FIFA World Cup 2026 highlights"
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{YOUTUBE_API_BASE}/search",
            params={
                "key": YOUTUBE_API_KEY,
                "q": query,
                "part": "snippet",
                "type": "video",
                "maxResults": max_results,
                "order": "relevance",
            },
            timeout=10,
        )
        resp.raise_for_status()
        items = resp.json().get("items", [])

    return [
        {
            "video_id": item["id"]["videoId"],
            "title": item["snippet"]["title"],
            "description": item["snippet"]["description"][:300],
            "channel": item["snippet"]["channelTitle"],
        }
        for item in items
    ]


def extract_transcript(video_id: str) -> Optional[str]:
    """
    Use yt-dlp to extract auto-generated subtitles/transcript.
    Falls back to description if unavailable.
    """
    try:
        result = subprocess.run(
            [
                "yt-dlp",
                "--skip-download",
                "--write-auto-sub",
                "--sub-lang", "en",
                "--sub-format", "vtt",
                "--output", f"/tmp/ytvtt/{video_id}",
                f"https://www.youtube.com/watch?v={video_id}",
            ],
            capture_output=True, text=True, timeout=60
        )
        vtt_path = Path(f"/tmp/ytvtt/{video_id}.en.vtt")
        if vtt_path.exists():
            raw = vtt_path.read_text(errors="ignore")
            # Strip VTT formatting — keep only spoken lines
            lines = [
                line.strip() for line in raw.splitlines()
                if line.strip()
                and not line.startswith("WEBVTT")
                and not line.startswith("NOTE")
                and "-->" not in line
                and not line.strip().isdigit()
            ]
            return " ".join(lines)[:8000]
    except Exception as e:
        print(f"[video] transcript extraction failed for {video_id}: {e}")
    return None


def analyze_team_form_from_transcripts(
    team_name: str,
    transcripts: list[str],
    title_descriptions: list[str],
) -> str:
    """
    Use Claude to extract tactical insights and player form from video transcripts.
    Returns a concise narrative suitable as seed material for match prediction.
    """
    if not transcripts and not title_descriptions:
        return ""

    content_block = "\n\n---\n\n".join(
        [f"Video {i+1}: {td}\nTranscript: {tr}"
         for i, (td, tr) in enumerate(zip(title_descriptions, transcripts or [""] * len(title_descriptions)))]
    )

    prompt = f"""Analyze these YouTube video transcripts and titles about {team_name} at FIFA World Cup 2026.

{content_block}

Extract and summarize:
1. Current form and momentum (wins/losses, confidence level)
2. Key players in form (top performers, goal scorers, assists)
3. Tactical patterns (playing style, pressing, defensive shape)
4. Injury/availability concerns mentioned
5. Notable strengths and weaknesses observed

Be concise and factual. Focus only on information useful for predicting match outcomes."""

    response = _anthropic_client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=600,
        system="You are a football analyst extracting tactical insights from video content.",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


async def get_team_video_insights(
    team_name: str,
    cache_dir: str = "./data/cache",
) -> str:
    """Full pipeline: search → transcript → Claude analysis. Returns insight string."""
    cache_path = Path(cache_dir) / f"video_{team_name.replace(' ', '_').lower()}.txt"
    if cache_path.exists():
        return cache_path.read_text()

    videos = await search_team_highlights(team_name, max_results=3)
    if not videos:
        return ""

    title_descs = [f"{v['title']} - {v['description']}" for v in videos]
    transcripts = []
    for v in videos:
        transcript = extract_transcript(v["video_id"])
        transcripts.append(transcript or "")

    insight = analyze_team_form_from_transcripts(team_name, transcripts, title_descs)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(insight)
    return insight
