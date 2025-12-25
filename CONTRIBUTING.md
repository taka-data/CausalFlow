# Contributing to CausalFlow

Thank you for your interest in contributing to CausalFlow!

## Development Workflow

1.  **Create a Feature Branch**:
    ```bash
    git checkout -b feat/your-feature-name
    ```

2.  **Rust Development**:
    - Ensure code is formatted: `cargo fmt`
    - Check for common issues: `cargo clippy`
    - Run tests: `cargo test`

3.  **Python Development**:
    - Rebuild the extension after changes:
      ```bash
      cd py-causalflow
      maturin develop
      ```
    - Verify with the example script:
      ```bash
      python ../example_causal_analysis.py
      ```

4.  **Submit a Pull Request**:
    - Push your branch to GitHub.
    - Open a PR against the `main` branch.
    - Fill out the PR template.
    - Wait for CI checks to pass.

## CI Checks

Every PR triggers a GitHub Action that runs:
- `cargo fmt` check
- `clippy` linting
- `cargo test`
- `maturin build`

Please ensure these pass before requesting a review.
