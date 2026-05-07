# ══════════════════════════════════════════════════════════════════════════════
# Makefile — IDBI Gift Voucher Recommendation Engine
# Mirrors the Makefile commands referenced in the project report.
# ══════════════════════════════════════════════════════════════════════════════

.PHONY: setup data lint test train destroy clean all

# ── 1. UV Environment Setup ───────────────────────────────────────────────────
setup:
	@echo "═══ Setting up UV environment ═══"
	uv venv .venv
	@echo "  ✓ Virtual environment created at .venv"
	uv init --no-workspace 2>/dev/null || true
	uv add -r requirements.txt
	uv sync
	@echo "  ✓ Dependencies installed and uv.lock generated"

# ── 2. Generate Dataset ───────────────────────────────────────────────────────
data:
	@echo "═══ Generating synthetic bank voucher dataset ═══"
	uv run python data/generate_dataset.py

# ── 3. Code Linting (Ruff — as per report Section 3.3) ───────────────────────
lint:
	@echo "═══ Ruff Linting ═══"
	uv run ruff --version
	uv run ruff check . --select F --ignore E402 --output-format grouped

# ── 4. Run Smoke Tests (CI Phase — The Gauntlet) ──────────────────────────────
test:
	@echo "═══ Smoke Tests (CI Phase) ═══"
	uv run pytest tests/test_smoke.py -v

# ── 5. Run All Tests (including post-training) ────────────────────────────────
test-all:
	@echo "═══ Full Test Suite ═══"
	uv run pytest tests/ -v

# ── 6. Train Pipeline ─────────────────────────────────────────────────────────
train:
	@echo "═══ Training Pipeline ═══"
	uv run python train_pipeline.py

# ── 7. Docker: Build Image ────────────────────────────────────────────────────
docker-build:
	@echo "═══ Building Docker image ═══"
	docker build -t recommendation-engine-idbi:latest .

# ── 8. Docker: Check container status ────────────────────────────────────────
docker-status:
	docker-compose ps

# ── 9. Docker: Run tests in container ────────────────────────────────────────
docker-test:
	@echo "═══ Smoke Tests in Container ═══"
	docker-compose run --rm app bash -c "uv run pytest tests/test_smoke.py -v"

# ── 10. Docker: Run training in container ────────────────────────────────────
docker-train:
	@echo "═══ Training Pipeline in Container ═══"
	docker-compose run --rm app bash -c "uv run python train_pipeline.py"

# ── 11. Post-Training Artifact Check ─────────────────────────────────────────
check-artifacts:
	@echo "═══ Post-Training Artifact Validation ═══"
	uv run pytest tests/post_training/ -v -k pkl

# ── 12. Destroy: Stop + Remove container (as per report CD post-flight) ───────
destroy:
	@echo "═══ Destroying container ═══"
	-docker stop recommendation-engine-idbi 2>/dev/null || true
	-docker rm recommendation-engine-idbi 2>/dev/null || true
	@echo "  ✓ Container destroyed — zero memory footprint"

# ── 13. Full Local Pipeline (No Docker) ──────────────────────────────────────
all: data lint test train check-artifacts
	@echo "═══ Full pipeline complete ═══"

# ── 14. Clean artifacts ───────────────────────────────────────────────────────
clean:
	@echo "═══ Cleaning artifacts ═══"
	rm -f model_artifacts/*.pkl
	rm -f logs/*.log
	rm -f *.db
	@echo "  ✓ Clean complete"
