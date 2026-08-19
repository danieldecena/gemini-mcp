#!/usr/bin/env python3
"""Gemini MCP server -- exposes Gemini to Claude Code as native tools.

Two tools, chosen for Gemini's actual edge over Claude (see the /gemini skill and
the reference_gemini_antigravity_cli memory for the full rationale):

  gemini_ask  -- offload a large-context read or capable bulk text task. Gemini's
                 ~1M-token window swallows inputs too big for Claude's context;
                 hand it the text, get back a compact result Claude reasons over.
  gemini_web  -- answer using live web context: Google Search grounding plus
                 optional URL reading. This is Gemini's strongest edge (Google's
                 index, first-party). Returns the answer with source citations.

Auth: reads GEMINI_API_KEY from the environment (inherited from the shell that
launched Claude Code, which sources ~/.secrets/env.zsh). The key is NEVER placed
in settings.json. If it is missing, the tools return a clear error string rather
than crashing the server.

Both tools are advisory: their output is a draft for Claude to verify, not
authority. Do not trust facts, decisions, or unreviewed code from them blindly.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from google import genai
from google.genai import types
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("gemini")

DEFAULT_MODEL = "gemini-2.5-flash"  # flash-lite is retired (404) for new keys


def parse_model_json(text: str) -> dict:
    """Load the first JSON object from a model reply, ignoring surrounding text.

    A URL-context call can't also enforce a response schema (the API 400s), so
    gemini_extract instructs the JSON shape in the prompt. Observed live, the
    model wraps the object in a markdown fence, prepends chatty prose, and/or adds
    a trailing note -- all in the same reply. Rather than strip each wrapper, find
    the first `{` and decode one complete object from there, discarding whatever
    precedes or follows it.
    """
    start = text.find("{")
    if start == -1:
        raise ValueError("no JSON object found in model reply")
    obj, _ = json.JSONDecoder().raw_decode(text, start)
    return obj


def _load_key_fallback() -> None:
    """If the launcher didn't pass GEMINI_API_KEY, read it from ~/.secrets/env.zsh.

    Keeps the key in its one secure place (mode 600, gitignored) instead of
    settings.json, and makes the server work regardless of how Claude Code scopes
    the MCP process environment.
    """
    if os.environ.get("GEMINI_API_KEY"):
        return
    try:
        for line in (Path.home() / ".secrets" / "env.zsh").read_text().splitlines():
            m = re.match(r"""\s*export\s+GEMINI_API_KEY=["']?([^"'\s]+)""", line)
            if m:
                os.environ["GEMINI_API_KEY"] = m.group(1)
                return
    except OSError:
        pass


_CLIENT: genai.Client | None = None


def _client() -> genai.Client:
    """Return one cached client; raises if GEMINI_API_KEY is unset (caught by callers).

    A fresh Client() per call trips google-genai's shared-transport lifecycle
    ("Cannot send a request, as the client has been closed"), so cache it.
    """
    global _CLIENT
    if not os.environ.get("GEMINI_API_KEY"):
        _load_key_fallback()
    if not os.environ.get("GEMINI_API_KEY"):
        raise RuntimeError("GEMINI_API_KEY not set and not found in ~/.secrets/env.zsh")
    if _CLIENT is None:
        _CLIENT = genai.Client()
    return _CLIENT


def _citations(resp) -> str:
    """Best-effort extraction of grounding source URLs; empty string if none."""
    try:
        meta = resp.candidates[0].grounding_metadata
        seen, lines = set(), []
        for chunk in getattr(meta, "grounding_chunks", None) or []:
            web = getattr(chunk, "web", None)
            if web and web.uri and web.uri not in seen:
                seen.add(web.uri)
                lines.append(f"- {web.title or web.uri}: {web.uri}")
        return "\n".join(lines)
    except Exception:
        return ""


@mcp.tool()
def gemini_ask(prompt: str, model: str = DEFAULT_MODEL) -> str:
    """Offload a large-context read or capable bulk text task to Gemini.

    Use for inputs too large or costly to pull fully into Claude's context (long
    docs, transcripts, corpora), or advisory bulk work (summarize/classify/rewrite
    many items) that is beyond the local model but not worth Opus. Paste the full
    text into `prompt` -- the call is stateless and sees no local files.

    Not for: facts you'll trust unverified, decisions, security-adjacent text, or
    code you'd ship unreviewed. Treat the reply as a draft to check.

    Args:
        prompt: the full instruction plus any content to operate on.
        model: Gemini model id; default gemini-2.5-flash (use gemini-2.5-pro for
            harder reasoning).
    """
    try:
        r = _client().models.generate_content(model=model, contents=prompt)
        return r.text or "(empty response)"
    except Exception as e:
        return f"error: {type(e).__name__}: {e}"


@mcp.tool()
def gemini_web(
    prompt: str, urls: list[str] | None = None, model: str = DEFAULT_MODEL
) -> str:
    """Answer using live web context: Google Search grounding + optional URL reading.

    Gemini's strongest edge -- use it to get current information, research a topic
    with citations, or read and digest specific pages. Pass the pages to read in
    `urls`; Gemini fetches them itself (no scraping needed). The reply includes a
    Sources list when grounding returns citations.

    Not for: anything you'll act on without Claude verifying it. Advisory context,
    not authority.

    Args:
        prompt: the question or extraction instruction.
        urls: optional specific pages for Gemini to read.
        model: Gemini model id; default gemini-2.5-flash.
    """
    contents = prompt
    if urls:
        contents = prompt + "\n\nRead these URLs:\n" + "\n".join(urls)
    try:
        r = _client().models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(
                tools=[
                    types.Tool(google_search=types.GoogleSearch()),
                    types.Tool(url_context=types.UrlContext()),
                ],
            ),
        )
        out = r.text or "(empty response)"
        cites = _citations(r)
        return out + (f"\n\nSources:\n{cites}" if cites else "")
    except Exception as e:
        return f"error: {type(e).__name__}: {e}"


@mcp.tool()
def gemini_extract(url: str, fields: list[str], model: str = DEFAULT_MODEL) -> dict:
    """Extract named fields from a web page as structured JSON via Gemini.

    Gemini reads the page itself (the URL-context tool) and returns just the
    fields you asked for -- no scraping. Use for pulling structured data out of a
    page (marketing copy, product specs, listing fields) in one call.

    The URL-context tool and a strict response schema can't combine (the API
    400s), so the JSON shape is instructed in the prompt and the reply is parsed
    (fence-tolerant). The result is the requested keys plus `url`.

    Not for: data you'll act on without Claude verifying it. Advisory, not
    authority.

    Args:
        url: the page to read.
        fields: the keys to extract (e.g. ["headline", "primary_cta"]).
        model: Gemini model id; default gemini-2.5-flash.
    """
    field_list = ", ".join(fields)
    prompt = (
        f"Read the page at {url} and extract these fields: {field_list}.\n"
        "Return ONLY a JSON object with exactly those keys. Use null for anything "
        "genuinely absent. Do not invent. Output JSON only, no prose."
    )
    try:
        r = _client().models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[types.Tool(url_context=types.UrlContext())],
            ),
        )
        data = parse_model_json(r.text)
    except Exception as e:
        return {"url": url, "error": f"{type(e).__name__}: {e}"}
    data["url"] = url
    return data


if __name__ == "__main__":
    mcp.run()
