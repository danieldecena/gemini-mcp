"""Tests for the Gemini MCP server's pure logic (no network)."""

import server


def test_parses_json_wrapped_in_a_fenced_code_block():
    reply = '```json\n{"headline": "Ship faster"}\n```'
    assert server.parse_model_json(reply) == {"headline": "Ship faster"}


def test_parses_json_followed_by_a_trailing_note_without_a_fence():
    # Observed live: gemini-2.5-flash returns the object then keeps talking.
    reply = '{"headline": "Ship faster"}\n\nLet me know if you need more fields.'
    assert server.parse_model_json(reply) == {"headline": "Ship faster"}


def test_parses_json_with_chatty_prose_before_an_unfenced_object():
    # Observed live: "I have browsed the page. ... I will now present it: {...}"
    reply = 'I have browsed the page and will present it: {"headline": "Ship faster"}'
    assert server.parse_model_json(reply) == {"headline": "Ship faster"}


def test_parses_json_when_the_model_adds_prose_before_the_fence():
    reply = 'Here is the data you asked for:\n```json\n{"cta": "Start free"}\n```'
    assert server.parse_model_json(reply) == {"cta": "Start free"}


class _FakeResponse:
    def __init__(self, text):
        self.text = text


class _FakeClient:
    """Records the call and returns a canned fenced-JSON reply."""

    def __init__(self, text):
        self._text = text
        self.last_contents = None
        self.models = self

    def generate_content(self, *, model, contents, config=None):
        self.last_contents = contents
        return _FakeResponse(self._text)


def test_gemini_extract_parses_the_reply_and_attaches_the_source_url(monkeypatch):
    fake = _FakeClient('```json\n{"headline": "Ship faster", "cta": "Start"}\n```')
    monkeypatch.setattr(server, "_client", lambda: fake)

    result = server.gemini_extract("https://acme.test", ["headline", "cta"])

    assert result == {
        "headline": "Ship faster",
        "cta": "Start",
        "url": "https://acme.test",
    }
    # the requested fields and the URL must reach the model in the prompt
    assert (
        "headline" in fake.last_contents and "https://acme.test" in fake.last_contents
    )


def test_gemini_extract_returns_an_error_dict_when_the_reply_is_not_json(monkeypatch):
    fake = _FakeClient("Sorry, I could not read that page.")
    monkeypatch.setattr(server, "_client", lambda: fake)

    result = server.gemini_extract("https://acme.test", ["headline"])

    assert result["url"] == "https://acme.test"
    assert "error" in result
