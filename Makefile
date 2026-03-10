run:
	docker compose up -d

build:
	docker compose build

test:
	docker compose run app pytest

insert-songs:
	docker compose run --rm insert

fingerprint:
	docker compose run --rm fingerprint

evaluate:
	docker compose run --rm evaluate

logs:
	docker compose logs -f app

down:
	docker compose down

redis:
	docker compose up redis -d
