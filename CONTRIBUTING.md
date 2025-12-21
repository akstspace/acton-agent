# Contributing to Acton Agent

Thank you for your interest in contributing to Acton Agent! 

## Current Status

⚠️ **Important**: Acton Agent is currently a **personal experimental project**. While it's open source under the MIT license, active contributions are not being sought at this time.

## How You Can Help

Even though active development contributions aren't being accepted, you can still help:

### Report Bugs

If you find a bug, please [open an issue](https://github.com/akstspace/acton-agent/issues) with:

- **Description**: What happened vs. what you expected
- **Steps to reproduce**: Minimal code example
- **Environment**: Python version, OS, package version
- **Error messages**: Full traceback if applicable

**Example:**
```markdown
## Bug: Agent crashes on empty tool result

**Description:** Agent raises UnboundLocalError when tool returns empty string

**Steps to reproduce:**
```python
def empty_tool():
    return ""

tool = FunctionTool(name="test", description="Test", func=empty_tool, schema={...})
agent.register_tool(tool)
agent.run("Use the test tool")
```

**Error:**
```
UnboundLocalError: local variable 'result' referenced before assignment
```

**Environment:**
- Python 3.11
- acton-agent 0.0.9
- macOS 14.0
```

### Suggest Features

Have an idea? [Open an issue](https://github.com/akstspace/acton-agent/issues) with the `enhancement` label:

- **Use case**: What problem does it solve?
- **Proposal**: How should it work?
- **Alternatives**: What other approaches did you consider?
- **Examples**: Code examples of proposed API

### Improve Documentation

Found a typo or unclear explanation? Documentation improvements are welcome:

1. Fork the repository
2. Make your changes to the docs
3. Submit a pull request

Focus on:
- Fixing typos and grammar
- Clarifying confusing sections
- Adding missing examples
- Improving code comments

### Share Your Experience

- **Use cases**: How are you using Acton Agent?
- **Integrations**: What tools/APIs have you integrated?
- **Patterns**: Useful patterns you've discovered?

Share in [GitHub Discussions](https://github.com/akstspace/acton-agent/discussions).

## Development Setup

If you want to work on Acton Agent locally:

### 1. Clone the Repository

```bash
git clone https://github.com/akstspace/acton-agent.git
cd acton-agent
```

### 2. Install Development Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in editable mode with dev dependencies
pip install -e ".[dev]"
```

### 3. Run Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=acton_agent --cov-report=html

# Run specific test file
pytest tests/test_agent.py
```

### 4. Code Quality

```bash
# Format code
ruff format .

# Lint code
ruff check .

# Type checking
mypy acton_agent
```

### 5. Pre-commit Hooks

```bash
# Install pre-commit hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

## Code Style

Acton Agent follows these conventions:

### Python Style

- **Formatting**: Automated with `ruff format`
- **Line length**: 100 characters
- **Imports**: Sorted with `isort` profile
- **Type hints**: Prefer type annotations
- **Docstrings**: Google style

**Example:**
```python
def my_function(param1: str, param2: int = 0) -> bool:
    """
    Do something useful.
    
    Args:
        param1: Description of param1
        param2: Description of param2. Defaults to 0.
    
    Returns:
        True if successful, False otherwise.
    
    Raises:
        ValueError: If param1 is empty
    """
    if not param1:
        raise ValueError("param1 cannot be empty")
    
    return True
```

### Naming Conventions

- **Classes**: `PascalCase` (e.g., `Agent`, `FunctionTool`)
- **Functions/methods**: `snake_case` (e.g., `register_tool`, `run_stream`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `MAX_RETRIES`)
- **Private**: Leading underscore (e.g., `_internal_method`)

### Error Handling

```python
# Good: Specific exceptions
try:
    result = risky_operation()
except ValueError as e:
    logger.error(f"Invalid value: {e}")
    raise

# Avoid: Bare except
try:
    result = risky_operation()
except:  # ❌ Too broad
    pass
```

### Logging

Use `loguru` for logging:

```python
from loguru import logger

logger.debug("Detailed information for debugging")
logger.info("General information")
logger.warning("Warning message")
logger.error("Error message")
logger.success("Success message")
```

## Testing Guidelines

### Test Structure

```python
import pytest
from acton_agent import Agent

class TestAgent:
    """Tests for Agent class."""
    
    def test_initialization(self):
        """Test agent initializes correctly."""
        agent = Agent(llm_client=mock_client)
        assert agent is not None
    
    def test_run_simple_query(self):
        """Test agent can handle simple query."""
        agent = Agent(llm_client=mock_client)
        result = agent.run("Test query")
        assert result == "Expected response"
```

### Test Coverage

- Aim for >80% coverage
- Test edge cases
- Test error conditions
- Mock external dependencies

### Fixtures

Use pytest fixtures for common setup:

```python
@pytest.fixture
def mock_client():
    """Mock LLM client for testing."""
    return MockLLMClient(responses=["Test response"])

@pytest.fixture
def agent(mock_client):
    """Agent instance for testing."""
    return Agent(llm_client=mock_client)

def test_something(agent):
    # Use the agent fixture
    result = agent.run("Test")
    assert result
```

## Documentation

### Docstrings

All public functions, classes, and methods should have docstrings:

```python
class MyClass:
    """
    Brief description of the class.
    
    Longer description if needed, explaining purpose,
    usage, and any important details.
    
    Attributes:
        attribute1: Description of attribute1
        attribute2: Description of attribute2
    
    Example:
        ```python
        obj = MyClass(param="value")
        result = obj.method()
        ```
    """
    
    def method(self, param: str) -> bool:
        """
        Brief description of method.
        
        Args:
            param: Description of parameter
        
        Returns:
            Description of return value
        
        Raises:
            ValueError: When this exception is raised
        """
        pass
```

### README and Guides

- Use clear, simple language
- Include code examples
- Show expected output
- Link related sections

## Pull Request Process

If submitting a PR for documentation:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b docs/improve-examples`)
3. **Make** your changes
4. **Test** that examples work
5. **Commit** with clear messages
6. **Push** to your fork
7. **Create** a pull request

### PR Description

```markdown
## Changes

Brief description of what changed

## Motivation

Why this change is needed

## Testing

How you tested the changes

## Checklist

- [ ] Documentation is clear
- [ ] Examples run without errors
- [ ] No typos or grammar issues
- [ ] Links work correctly
```

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Questions?

- **Issues**: [github.com/akstspace/acton-agent/issues](https://github.com/akstspace/acton-agent/issues)
- **Discussions**: [github.com/akstspace/acton-agent/discussions](https://github.com/akstspace/acton-agent/discussions)

Thank you for your interest in making Acton Agent better! 🚀
