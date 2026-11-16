
default:
    @just --list

alias t   := test
alias f   := format
alias l   := lint
alias tv  := test-verbose
alias run := run-file


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

# Regenerate tier-2 anywhere level bank
gen-tier2 seed="0":
    uv run python -m opus_magnum_blackbox.gen_tier2 --seed {{seed}}

# Regenerate tier-2 per-class enumerated banks (tier2_single + tier2_handoff)
gen-tier2-classes seed="0" eval_percent="20":
    uv run python -m opus_magnum_blackbox.gen_tier2 --mode classes --seed {{seed}} --eval-percent {{eval_percent}}

# Train the baseline policy on tier-0 level (default overfit config)
train:
    uv run opus-magnum-blackbox train

# Run tier-1 curriculum training pipeline
train-tier1:
    uv run opus-magnum-blackbox run configs/tier1_simple_arms.toml

# Run tier-2 anywhere curriculum pipeline
train-tier2:
    uv run opus-magnum-blackbox run configs/tier2_anywhere.toml

# Run tier-2 curriculum-split s(single-arm bank, ring-stratified SIL)
train-tier2-single:
    uv run opus-magnum-blackbox run configs/tier2_single.toml

# Run tier-2 curriculum-split (handoff bank; fill [run] init_from first)
train-tier2-handoff:
    uv run opus-magnum-blackbox run configs/tier2_handoff.toml

# Run tier-2 curriculum-split capacity arm (bigger model, fresh)
train-tier2-single-big:
    uv run opus-magnum-blackbox run configs/tier2_single_big.toml

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
