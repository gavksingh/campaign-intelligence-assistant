.PHONY: setup seed run test lint docker-up docker-down clean web-setup web-dev

# ── Local development ────────────────────────────────────────────────

setup:
	python -m pip install -r requirements.txt
	@echo "Dependencies installed. Copy .env.example -> .env and add your OPENAI_API_KEY."

seed:
	python -m data.seed
	@echo "Database seeded with mock campaigns."

run:
	uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload

test:
	python -m pytest tests/ -v

lint:
	python -m ruff check app/ tests/
	python -m ruff format --check app/ tests/

format:
	python -m ruff format app/ tests/

# ── Next.js frontend ────────────────────────────────────────────────

web-setup:
	cd web && npm install

web-dev:
	cd web && npm run dev

web-build:
	cd web && npm run build

# ── Docker ───────────────────────────────────────────────────────────

docker-up:
	docker compose up -d --build

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f app

# ── Cleanup ──────────────────────────────────────────────────────────

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf .ruff_cache
