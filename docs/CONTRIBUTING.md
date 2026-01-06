# Contributing Guide

## Welcome!

Thank you for considering contributing to Yolo11_auto_train!

## Development Setup

### 1. Clone Repository
```bash
git clone <repo-url>
cd Yolo11_auto_train
```

### 2. Install Dependencies
```bash
# Development environment (includes all tools)
pip install -r requirements-dev.txt

# Or install in editable mode
pip install -e ".[dev,gui]"
```

### 3. Verify Installation
```bash
# Run tests
pytest

# Check code quality
ruff check src tests
```

## Development Workflow

### 1. Create Branch
```bash
git checkout -b feature/your-feature-name
```

### 2. Make Changes
- Write code
- Add tests
- Update documentation

### 3. Test
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=picture_tool --cov-report=html

# Run specific test
pytest tests/test_specific.py -v
```

### 4. Lint
```bash
# Check code quality
ruff check src tests

# Format code
ruff format src tests
```

### 5. Commit
```bash
git add .
git commit -m "feat: add new feature"
```

Follow conventional commit format:
- `feat:` new feature
- `fix:` bug fix
- `docs:` documentation
- `refactor:` code refactoring
- `test:` add tests

## Code Style

- Follow PEP 8
- Use type hints
- Maximum line length: 100
- Use ruff for linting/formatting

## Testing Guidelines

### Writing Tests
```python
def test_feature_basic_case(tmp_path):
    """Test basic functionality."""
    # Arrange
    config = {...}
    
    # Act
    result = function_under_test(config)
    
    # Assert
    assert result == expected
```

### Coverage Requirements
- New code: >80% coverage
- Critical paths: 100% coverage
- Overall target: 70%

## Documentation

### Docstrings
```python
def function(arg1: str, arg2: int) -> bool:
    """
    Brief description.
    
    Args:
        arg1: Description
        arg2: Description
        
    Returns:
        Description
        
    Raises:
        ValidationError: When...
    """
```

### Update Docs
- Add to `docs/` for major features
- Update `README.md` if API changes
- Add examples to `docs/guides/`

## Pull Request Process

1. Update tests
2. Update documentation
3. Ensure all tests pass
4. Run linting
5. Create PR with description
6. Wait for review
7. Address feedback
8. Merge after approval

## Project Structure

See `docs/ARCHITECTURE.md` for details.

## Questions?

Open an issue or discussion on GitHub.
