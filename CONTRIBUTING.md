# Contributing to PharmKG-DTI

Thank you for your interest in contributing to PharmKG-DTI! This document provides guidelines and instructions for contributing.

## Development Setup

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/pharmkg-dti.git
   cd pharmkg-dti
   ```

3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

5. Install development dependencies:
   ```bash
   pip install pytest black flake8 mypy
   ```

## Code Style

We use the following tools to maintain code quality:

- **Black**: Code formatting
- **flake8**: Linting
- **mypy**: Type checking (optional but recommended)

Before committing, run:
```bash
black src/ tests/
flake8 src/ tests/
```

## Testing

All new features should include tests. Run the test suite:

```bash
pytest tests/ -v
```

For coverage report:
```bash
pytest tests/ --cov=src --cov-report=html
```

## Pull Request Process

1. Create a new branch for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and add tests

3. Run tests and ensure they pass

4. Commit your changes:
   ```bash
   git commit -m "Add feature: your feature description"
   ```

5. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

6. Open a Pull Request on GitHub

## Commit Message Guidelines

- Use present tense ("Add feature" not "Added feature")
- Use imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit first line to 72 characters
- Reference issues and PRs when applicable

Example:
```
Add uncertainty estimation module

- Implement Monte Carlo Dropout
- Add ensemble disagreement metric
- Include calibration curve computation

Fixes #123
```

## Reporting Issues

When reporting issues, please include:

- Python version
- PyTorch version
- Operating system
- Full error message/traceback
- Minimal code to reproduce the issue

## Code Review Process

All submissions require review. We aim to respond to PRs within 48 hours.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Questions?

Feel free to open an issue for questions or join our discussions.

Thank you for contributing!
