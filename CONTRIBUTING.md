# Contributing to Lean GRPO

Thank you for your interest in contributing to Lean GRPO! This document provides guidelines for contributing.

## Development Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/lean-grpo.git
   cd lean-grpo
   ```

2. **Install with dev dependencies:**
   ```bash
   make install-dev
   ```

3. **Set up pre-commit hooks:**
   ```bash
   pre-commit install
   ```

## Development Workflow

1. **Create a branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**

3. **Run tests:**
   ```bash
   make test
   ```

4. **Run linters:**
   ```bash
   make lint
   make format
   ```

5. **Commit and push:**
   ```bash
   git add .
   git commit -m "feat: your feature description"
   git push origin feature/your-feature-name
   ```

6. **Create a Pull Request**

## Code Style

- Follow PEP 8
- Use type hints
- Write docstrings in Google style
- Keep functions focused and small
- Add tests for new features

## Testing

- Write tests for new functionality
- Maintain test coverage above 80%
- Use pytest for testing
- Mock external dependencies (Lean, API calls)

## Commit Messages

Follow conventional commits:

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `test:` Test changes
- `refactor:` Code refactoring
- `style:` Code style changes
- `chore:` Maintenance tasks

## Reporting Issues

When reporting issues, please include:

- Python version
- GPU/CUDA version
- Lean 4 version
- Full error traceback
- Minimal reproducible example

## Questions?

Feel free to open an issue for questions or join our discussions.
