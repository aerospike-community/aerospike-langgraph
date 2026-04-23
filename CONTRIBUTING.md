# Contributing to [PROJECT_NAME]

Thank you for your interest in contributing to this Aerospike project! We welcome contributions from the community.

## How to Contribute

### **Did you find a bug?**

- **Do not open up a GitHub issue if the bug is a security vulnerability**, and instead refer to our [security policy](SECURITY.md)

- If you're unable to find an open issue addressing the problem, be sure to include a **title and clear description**, as much relevant information as possible, and a **code sample** or an **executable test case** demonstrating the expected behavior that is not occurring.

### **Did you write a patch?**

- Open a new GitHub pull request with the patch.

- Ensure the PR description clearly describes the problem and solution. Include the relevant issue number if applicable.

## Development Setup

### Repo Tooling

Linting will be run on PRs; you can save yourself some time and annoyance by linting as you write.

If you use Visual Studio Code or a derivative, there are suggested extensions in the [.vscode](.vscode) directory.

### pre-commit hooks

This repo uses [`ruff`](https://docs.astral.sh/ruff/) for linting/formatting and [`mypy`](https://mypy.readthedocs.io/) for type checking, both wired up via [`pre-commit`](https://pre-commit.com/). After `uv sync`, install the git hooks once so they run automatically on every `git commit`:

```bash
uv run pre-commit install
```

To run all checks against the whole repo manually (same checks CI runs):

```bash
uv run pre-commit run --all-files
```

### Contributor

This project adheres to the Contributor Covenant [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## Questions?

Feel free to open an issue with your question.

## License

By contributing, you agree that your contributions will be licensed under the same license as the project (see [LICENSE](LICENSE)).
