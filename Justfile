
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

# Regenerate tier-1 simple-arms level bank
gen-tier1 seed="0":
    uv run python -m opus_magnum_blackbox.gen_tier1 --seed {{seed}}

# Train the baseline policy on tier-0 level (default overfit config)
train:
    uv run opus-magnum-blackbox train

# Run tier-1 curriculum training pipeline
train-tier1:
    uv run opus-magnum-blackbox run configs/tier1_simple_arms.toml

# Run the canonical tier-1 discovery recipe (BC seed + stratified self-imitation)
train-tier1-discovery:
    uv run opus-magnum-blackbox run configs/tier1_discovery.toml

# Run tier-1 autodiscovery sweep (k in 1..10; k=1 equals train-tier1, baseline not rerun)
train-tier1-ad k:
    mkdir -p runs
    date +"%Y-%m-%dT%H:%M:%S.%6N"
    sed 's/%@/{{k}}/g' configs/tier1_ad.toml > runs/tier1_ad{{k}}.toml
    uv run opus-magnum-blackbox run runs/tier1_ad{{k}}.toml

# Run a stage pipeline from a TOML config
run-file file:
    uv run opus-magnum-blackbox run {{file}}

# Profile run: per-iteration sim vs model attribution (prints summary table)
profile-config config="configs/profile_tier0.toml":
    uv run opus-magnum-blackbox run {{config}}


# ===============================
# Tests
# ===============================

# Run default test suite
test:
    uv run pytest

# Run full test suite including corpus-scale coverage
test-corpus:
    uv run pytest --corpus

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
