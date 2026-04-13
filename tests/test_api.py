import asyncio

import httpx

from src.app.main import create_app


def test_health_endpoint_returns_pipeline_state() -> None:
    async def run_check() -> httpx.Response:
        app = create_app()
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                return await client.get("/health")

    response = asyncio.run(run_check())

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["mode"] == "letters"
