# gemini-mcp

Stdio MCP exposing Gemini to Claude: `gemini_ask` (large-context offload) and `gemini_web` (Search grounding). Auth is `GEMINI_API_KEY` from the environment, never from this repo.

## Stack
Python 3, `google-genai`, FastMCP. Default model `gemini-2.5-flash`.

## Do not
- Put the API key in settings or this tree.
- Treat tool output as authority; it is a draft to verify.
