# Acton Agent

> ⚠️ **Experimental Project**: This is a personal project currently in an experimental phase. The API may change without notice, and features may be incomplete or unstable. Use at your own discretion.

Acton Agent is a lightweight, flexible LLM Agent Framework with tool execution capabilities. It enables you to build AI agents that can interact with external APIs, execute custom Python functions, and maintain conversation context - all with minimal configuration.

## Installation

### Basic Installation

```bash
pip install acton-agent
```

### Installation with Optional Dependencies

For OpenAI integration:
```bash
pip install acton-agent[openai]
```

For development (includes testing and linting tools):
```bash
pip install acton-agent[dev]
```

Install all optional dependencies:
```bash
pip install acton-agent[all]
```

### Requirements

- Python >= 3.8
- Core dependencies:
  - `pydantic >= 2.0.0`
  - `tenacity >= 8.0.0`
  - `loguru >= 0.7.0`
  - `requests >= 2.31.0`


## Examples

Check out the [examples](examples/) directory:

- [examples/requests_tool_example.py](examples/requests_tool_example.py) - API integration with RequestsTool
- [examples/function_tool_example.py](examples/function_tool_example.py) - Custom Python function tools
- [examples/custom_tool_example.py](examples/custom_tool_example.py) - Building custom tool classes

## API Documentation

For detailed API documentation, please refer to the docstrings in the source code or visit our [GitHub repository](https://github.com/akstspace/acton-agent).

## Additional Information

### Current Status

This project is in **experimental phase** and is primarily for personal use. The following should be considered:

- **API Stability**: The API may change between versions without notice
- **Production Readiness**: Not recommended for production use yet
- **Documentation**: Documentation is being actively developed
- **Testing**: Test coverage is being expanded

### Contributing

As this is a personal experimental project, contributions are not actively sought at this time. However, if you find bugs or have suggestions, feel free to open an issue on GitHub.

### License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

### Support

For questions or issues, please use the [GitHub Issues](https://github.com/akstspace/acton-agent/issues) page.

---

**Disclaimer**: This is an experimental personal project. Use it at your own risk. The author makes no guarantees about stability, security, or fitness for any particular purpose.
