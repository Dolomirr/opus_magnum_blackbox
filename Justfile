
default:
    @just --list

alias t  := test
alias f  := format
alias l  := lint
alias tv := test-verbose


# ===============================
# Opus Magnum related commands
# ===============================

# Open REPL without visualization
om-repl-headless:
    uv run opus-magnum repl --headless

# Run solution headlessly
om-run file steps="30":
    uv run opus-magnum run {{file}} --steps {{steps}}

# Validate solution file
om-validate file:
    uv run opus-magnum validate {{file}}


# ===============================
# Training
# ===============================

# Train the baseline policy on tier-0 level (default overfit config)
train:
    uv run opus-magnum-blackbox train

# Run a stage pipeline from a TOML config
run-file file:
    uv run opus-magnum-blackbox run {{file}}

# Profile run: per-iteration sim vs model attribution (prints summary table)
profile-config config="configs/profile_tier0.toml":
    uv run opus-magnum-blackbox run {{config}}


# ===============================
# Tests
# ===============================

# Run test suite
test:
    uv run pytest

# Run test suite with verbose logging
test-verbose:
    uv run pytest -vv

# Run a single test file
test-file file:
    uv run pytest {{file}}

# Lint
lint:
    uv run ruff check --fix .

# Print only linter errors
lint-check-only:
    uv run ruff check src --output-format=concise 2>&1

# Format
format:
    uv run ruff format .

# Check with lsp
check:
    uv run ty check src

# Full code check
full: check lint test
