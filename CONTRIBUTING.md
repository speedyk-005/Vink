# Contributing to Vink

Hey! Thanks for thinking about contributing. Bug fixes, features, docs — all welcome.

## Getting Started

1. **Clone the repository:**

    ```bash
git clone https://github.com/speedyk-005/vinkra.git
cd vinkra
    ```

2. **Install dependencies:**

    1. **System dependencies** (Linux only — required for building `rii`):

    ```bash
    # Debian/Ubuntu
    sudo apt-get install python3-dev

    # RedHat/Fedora/CentOS
    sudo dnf install python3-devel -y

    # CentOS 7 and older
    sudo yum install python3-devel
    ```

    2. **Special dependencies** (required only on Android/Termux):

    ```bash
    pkg install -y tur-repo
    pkg install python-scipy
    ```

    3. **Python dependencies:**

    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    pip install -e ".[dev]"
    ```

<!-- 4. **Run the demo** to verify everything works:

    ```bash
    python demo_poc.py
    ``` -->

## Making Changes

1. **Create a new branch:** Use descriptive names like `feature/my-feature-branch` or `bugfix/issue-number-description`

    ```bash
    git checkout -b feature/my-feature-branch
    ```

2. **Write your code:** Follow the style guide below.

3. **Test:** Run `pytest` and make sure everything passes.

    ```bash
    pytest
    ```

4. **Format and lint:** Run `ruff format && ruff check --fix` before committing.

5. **Build documentation:**

    ```bash
    pip install -e ".[dev]"
    
    # Generate API_REFERENCES.md from docstrings
    python -m python_docstring_markdown ./src/vinkra API_REFERENCES.md
    
    # Generate/update README TOC (GitHub-compatible)
    npx doctoc --github README.md
    ```

## Pull Request Template

### Summary

Brief description of what this PR accomplishes.

### Changes

- Specific changes made
- Problems solved
- Impact on existing functionality

### Testing

- Tests added or modified
- Manual testing performed

### Related Issues

- Fixes #issue-number
  
## Submitting a Pull Request

Not sure about something? Open an issue first — happy to chat before you dive in.

- Descriptive title
- Summary of changes and why
- Link related issues ("Fixes #123")
- Ensure tests pass

## Code of Conduct

We're all adults here trying to build cool software together. Be nice, don't be a jerk, respect different opinions, and remember that behind every GitHub profile is a human being who probably has better things to do than deal with your nonsense. If you can't contribute without making the experience miserable for others, maybe try contributing to a different project instead. We're here to code, not to psychoanalyze each other's life choices. Keep it professional, keep it civil, and we'll all get along just fine.
