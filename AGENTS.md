# AGENTS.md

Context file for AI agents working on cohere-python.

**Dual Format**: This file combines Category A (Operations Manual) and Category B (Context Guide) for comprehensive agent guidance.

**Domain Detected:** Ml / Training (Based on codebase patterns)

## Project Overview

cohere-python is a Python project using Python (poetry).

**Key Info:**
- **Primary Language:** Python
- **Build System:** Python (poetry)
- **Test Framework:** pytest
- **Total Files:** 392
- **Test Files:** 15
- **AI Readiness Score:** 75/100 (AI-Native-Plus)

---

## 🚨 AI Policy & Operations

Extracted from CONTRIBUTING.md - operational constraints and procedures.

### AI Policy

- Thanks for your interest in contributing to this SDK! This document provides guidelines for contributing to the project.
- 3. Follow the [Fern contributing guidelines](https://github.com/fern-api/fern/blob/main/CONTRIBUTING.md)
- This project uses automated code formatting and linting. Run `poetry run ruff format .` and `poetry run ruff check .` before committing to ensure your code meets the project's style guidelines.

### Key Requirements

- If you need to customize the SDK, you have two options:

### Development Procedures

- Install the project dependencies:
- poetry install
- Build the project:
- poetry build
- Run the test suite:



## 🧠 Machine Learning Architecture

This is a machine learning or model training system.

### Key Components

- **Data Pipeline:** Data loading, preprocessing, augmentation
- **Model Definition:** Architecture, hyperparameters, checkpoints
- **Training Loop:** Loss calculation, gradient updates, validation
- **Inference:** Model predictions, batch processing, latency optimization
- **Evaluation:** Metrics, benchmarks, comparison to baselines

### Critical Areas

1. **Data Leakage:** Ensure train/test/validation splits are isolated
2. **Reproducibility:** Set random seeds; version datasets and models
3. **Resource Management:** Monitor memory, GPU usage during training
4. **Versioning:** Track model checkpoints, hyperparameters, and results
5. **Evaluation Rigor:** Use proper metrics; avoid optimizing to test set

### Testing Strategy

- **Data Pipeline Tests:** Verify shape, type, and value ranges
- **Model Tests:** Check predictions with synthetic/known inputs
- **Training Tests:** Verify loss decreases on toy datasets
- **Inference Tests:** Check latency and memory usage
- **Regression Tests:** Compare results against baseline models



### Detected Frameworks

| Framework | Version | Detection Type |
|-----------|---------|-----------------|
| pydantic | 1.9.2 | Direct import |
| requests | 2.0.0 | Direct import |
| unittest | unknown | Direct import |



## 🏗️ Architecture & Context Guide

This section provides architectural context and agent-understanding for the codebase.

### Prerequisites

- **Python:** 3.9+ (or applicable language version)
- **Package Manager:** pip or uv
- **Test Runner:** pytest



### Project Structure

```
cohere-python/
├── pyproject.toml
├── src/                  # Source code
├── tests/                # Test suite (15 files)
└── README.md             # Project documentation
```

### Architecture Overview

#### Key Components
- **Main Entry:** Standard layout
- **Test Suite:** 15 test files
- **Build Configuration:** pyproject.toml

#### Design Principles

1. **Modularity** - Code organized by functionality with clear separation of concerns
2. **Testability** - Comprehensive test coverage across critical paths
3. **Clarity** - Explicit naming and structure for AI agent understanding
4. **Consistency** - Uniform patterns and conventions throughout codebase
5. **Maintainability** - Well-documented code with clear intent

### Directory Map

| Directory | Purpose |
|-----------|----------|
| `src/` | Source code |
| `tests/` | Test suite |


### Development Workflow

#### Initial Setup

```bash
git clone https://github.com/cohere-ai/cohere-python
cd cohere-python
pip install -e .
# or
uv sync --all-groups
```

#### Development Commands

**Running Tests:**
```bash
pytest                    # Run all tests
pytest tests/             # Run specific test directory
pytest -v                 # Verbose output with test names
pytest -x                 # Stop on first failure
coverage run -m pytest && coverage report  # With coverage report
```

#### Code Quality
```bash
ruff check .              # Lint with ruff
ruff format .             # Format code
mypy .                    # Type checking (if configured)
```

### Code Style & Conventions

- **Naming:** Use snake_case for functions and variables
- **Type Hints:** Yes (strongly encouraged)
- **Error Handling:** Yes - handle errors at boundaries; let exceptions propagate when another layer owns recovery
- **Logging:** Yes
- **Testing:** Yes - write tests alongside code changes

### Testing Strategy

**Framework:** pytest
**Test Files:** 15 found

Before committing:
1. Run the full test suite: `pytest`
2. Ensure all tests pass
3. Check type hints: `mypy .`
4. Format code: `ruff format .`

### Writing Documentation

When updating docs:
1. Always include explanatory text before code snippets
2. Describe *why* and *what* before showing *how*
3. Keep sections focused on a single concept
4. Use clear, concrete examples

### Contributing Guidelines

This project has a detailed contribution guide at **`CONTRIBUTING.md`**.

**Key Requirements:**
- Review the contribution guide for all requirements
- Follow established patterns in the codebase
- Ensure alignment with project's contribution policies

### Common Patterns

When contributing to this project:
1. Read existing code in the area you're modifying
2. Follow the established patterns and style
3. Write tests for new functionality
4. Use clear, descriptive variable and function names
5. Add docstrings for public APIs
6. Update tests when changing behavior

### What We Value

✅ Well-tested code with clear intent
✅ Consistent code style and naming conventions
✅ Code that is easy for AI agents to understand
✅ Clear, descriptive commit messages
✅ Modular, reusable components
✅ Comprehensive documentation

### What We Avoid

❌ Large functions doing multiple things
❌ Commented-out dead code
❌ Inconsistent naming or patterns
❌ Unclear error messages
❌ Unexplained magic numbers or strings
❌ Skipped tests or test TODOs

### AI Readiness Dimensions (Scoring)

This project is evaluated across 8 dimensions:

1. **Architecture** (0/100) - Code organization and modularity
2. **Testing** (15/100) - Test coverage and quality
3. **Dependencies** (12/100) - Dependency management
4. **Conventions** (10/100) - Consistent patterns
5. **Entry Points** (0/100) - Clear main/start locations
6. **Security** (15/100) - Input validation and error handling
7. **Build** (10/100) - Clear build/setup instructions
8. **Documentation** (8/100) - Code and project documentation

### Next Steps

Before making changes:
1. Read relevant source files to understand the existing code
2. Look at existing tests for similar functionality
3. Follow the patterns you see in the codebase
4. Write tests for your changes
5. Run `pytest` to verify nothing breaks
6. Run code quality checks: `ruff check . && mypy .`
7. Format your code: `ruff format .`

---

*Generated by Braxis - keeping AI agents in sync with your code*
