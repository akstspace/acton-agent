"""
Tests for streaming functionality.
"""

from toolio_agent.agent.agent_stream import (
    StreamingResponseParser,
    extract_json_from_markdown,
    parse_partial_json,
)


class TestParsePartialJson:
    """Tests for parse_partial_json function."""

    def test_parse_complete_json(self):
        """Test parsing complete JSON."""
        json_str = '{"key": "value", "number": 42}'
        result = parse_partial_json(json_str)
        assert result == {"key": "value", "number": 42}

    def test_parse_partial_json_incomplete_object(self):
        """Test parsing incomplete JSON object."""
        json_str = '{"key": "value", "other":'
        result = parse_partial_json(json_str)
        # Should return what it can parse
        assert isinstance(result, dict)

    def test_parse_empty_string(self):
        """Test parsing empty string."""
        result = parse_partial_json("")
        assert result == {}

    def test_parse_incomplete_string_value(self):
        """Test parsing JSON with incomplete string value."""
        json_str = '{"key": "incomplete'
        result = parse_partial_json(json_str)
        assert isinstance(result, dict)

    def test_parse_json_array(self):
        """Test parsing JSON array."""
        json_str = '["item1", "item2", "item3"]'
        result = parse_partial_json(json_str)
        assert result == ["item1", "item2", "item3"]


class TestExtractJsonFromMarkdown:
    """Tests for extract_json_from_markdown function."""

    def test_extract_with_json_marker(self):
        """Test extracting JSON with json marker."""
        text = """Some text
```json
{"key": "value"}
```
More text"""

        result = extract_json_from_markdown(text)
        assert result == '{"key": "value"}'

    def test_extract_without_json_marker(self):
        """Test extracting JSON with generic code block."""
        text = """```
{"key": "value"}
```"""

        result = extract_json_from_markdown(text)
        assert result == '{"key": "value"}'

    def test_extract_no_code_block(self):
        """Test when no code block present."""
        text = '{"key": "value"}'
        result = extract_json_from_markdown(text)
        assert result is None

    def test_extract_inline_code_block(self):
        """Test extracting from inline code block."""
        text = '```json{"key": "value"}```'
        result = extract_json_from_markdown(text)
        # Should extract something
        assert result is not None


class TestStreamingResponseParser:
    """Tests for StreamingResponseParser."""

    def test_parser_initialization(self):
        """Test parser initialization."""
        parser = StreamingResponseParser()
        assert parser.buffer is not None
        assert parser.json_content == ""
        assert parser.last_parsed == {}

    def test_push_chunks(self):
        """Test pushing chunks to parser."""
        parser = StreamingResponseParser()

        # Push chunks that form a JSON block
        parser.push("```json\n")
        parser.push('{"thought": "test"')
        result = parser.push("}")

        # Should eventually parse something
        assert parser.get_current() is not None or result is not None

    def test_reset_parser(self):
        """Test resetting parser state."""
        parser = StreamingResponseParser()

        parser.push("```json\n")
        parser.push('{"key": "value"}')

        # Reset
        parser.reset()

        # Should be back to initial state
        assert parser.json_content == ""
        assert parser.last_parsed == {}

    def test_get_current(self):
        """Test getting current parsed state."""
        parser = StreamingResponseParser()

        current = parser.get_current()
        assert current == {}

        # After pushing some data
        parser.push('```json\n{"test": true}\n```')
        current = parser.get_current()
        # Should have parsed something
        assert isinstance(current, dict)
