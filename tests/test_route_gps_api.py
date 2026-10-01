import pytest
from fastapi.testclient import TestClient

from app import main


class FakeOSRM:
    async def get_routes(self, request):
        return [
            {
                "distance_km": 12.5,
                "duration_seconds": 1080,
                "geometry": [
                    {"lat": 30.7333, "lon": 76.7794, "altitude_m": 350},
                    {"lat": 30.8333, "lon": 76.8794, "altitude_m": 320},
                ],
            },
            {
                "distance_km": 15.2,
                "duration_seconds": 1400,
                "geometry": [
                    {"lat": 30.7333, "lon": 76.7794},
                    {"lat": 30.8000, "lon": 76.9000},
                    {"lat": 30.8333, "lon": 76.8794},
                ],
            },
        ]


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "route_provider", FakeOSRM())
    monkeypatch.setattr(
        main,
        "route_store",
        main.RouteStore(tmp_path / "routes.json"),
    )
    main.simulations.clear()
    with TestClient(main.app) as test_client:
        yield test_client
    main.simulations.clear()


def route_request():
    return {
        "origin": {"lat": 30.7333, "lon": 76.7794},
        "destination": {"lat": 30.8333, "lon": 76.8794},
        "vehicle": {
            "truck_type": "HCV",
            "gvw_kg": 18000,
            "height_m": 4.2,
            "width_m": 2.5,
            "axle_count": 6,
        },
    }


def test_generate_route_alternatives_and_persist_them(client, tmp_path):
    response = client.post("/api/v1/routes", json=route_request())

    assert response.status_code == 201
    routes = response.json()["routes"]
    assert len(routes) == 2
    assert routes[0]["distance_km"] == 12.5
    assert routes[0]["geometry"][0]["lat"] == 30.7333
    assert routes[0]["vehicle_constraints_verified"] is False
    assert routes[0]["vehicle"]["gvw_kg"] == 18000
    assert len(main.RouteStore(tmp_path / "routes.json").list_all()) == 2


def test_simulation_moves_on_saved_route_and_returns_history(client):
    route = client.post("/api/v1/routes", json=route_request()).json()["routes"][0]
    started = client.post(
        "/api/v1/simulations",
        json={
            "vehicle_id": "TRUCK-001",
            "route_id": route["route_id"],
            "speed_kmh": 60,
            "interval_seconds": 30,
            "auto_start": False,
        },
    )

    assert started.status_code == 201
    ticked = client.post(
        "/api/v1/simulations/TRUCK-001/tick",
        json={"advance_seconds": 30},
    )
    assert ticked.status_code == 200
    assert ticked.json()["lon"] > 76.7794
    assert ticked.json()["route_progress_percent"] > 0
    assert len(
        client.get("/api/v1/simulations/TRUCK-001/history").json()
    ) == 2


def test_invalid_route_coordinates_are_rejected(client):
    payload = route_request()
    payload["origin"]["lat"] = 120

    response = client.post("/api/v1/routes", json=payload)

    assert response.status_code == 422


def test_existing_optimizer_endpoint_still_responds(client):
    response = client.post(
        "/api/v1/optimize-route",
        json={
            "shipment_id": "SHIP-001",
            "constraints": {"gvw_kg": 25000},
            "origin": {"lat": 30.7046, "lon": 76.8010},
            "destination": {"lat": 17.6868, "lon": 83.2185},
            "cargo": {"type": "general", "weight_kg": 12500, "value": 100000},
        },
    )

    assert response.status_code == 200
    assert response.json()["shipment_id"] == "SHIP-001"


def test_missing_saved_route_returns_not_found(client):
    response = client.post(
        "/api/v1/simulations",
        json={
            "vehicle_id": "TRUCK-404",
            "route_id": "missing",
            "speed_kmh": 40,
        },
    )

    assert response.status_code == 404


def test_osrm_failure_returns_bad_gateway(client, monkeypatch):
    class BrokenOSRM:
        async def get_routes(self, request):
            raise RuntimeError("OSRM route request failed")

    monkeypatch.setattr(main, "route_provider", BrokenOSRM())
    response = client.post("/api/v1/routes", json=route_request())

    assert response.status_code == 502


def test_osrm_adapter_maps_geojson_coordinates(client, monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "code": "Ok",
                "routes": [
                    {
                        "distance": 12500,
                        "duration": 1080,
                        "geometry": {
                            "coordinates": [[76.7794, 30.7333], [76.8794, 30.8333]]
                        },
                    }
                ],
            }

    class FakeAsyncClient:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        async def get(self, url, params):
            assert url.endswith("/route/v1/driving/76.7794,30.7333;76.8794,30.8333")
            assert params["alternatives"] == "true"
            assert params["geometries"] == "geojson"
            return FakeResponse()

    monkeypatch.setattr(main, "route_provider", main.OSRMClient())
    monkeypatch.setattr(main.httpx, "AsyncClient", FakeAsyncClient)

    response = client.post("/api/v1/routes", json=route_request())

    assert response.status_code == 201
    point = response.json()["routes"][0]["geometry"][0]
    assert point == {"lat": 30.7333, "lon": 76.7794, "altitude_m": None}


def test_simulation_pause_resume_stop_and_duplicate_guard(client):
    route = client.post("/api/v1/routes", json=route_request()).json()["routes"][0]
    payload = {
        "vehicle_id": "TRUCK-PAUSE",
        "route_id": route["route_id"],
        "speed_kmh": 40,
        "auto_start": False,
    }

    started = client.post("/api/v1/simulations", json=payload)
    assert started.status_code == 201
    assert started.json()["status"] == "PAUSED"
    assert client.post("/api/v1/simulations", json=payload).status_code == 409
    assert client.post("/api/v1/simulations/TRUCK-PAUSE/resume").json()["status"] == "RUNNING"
    assert client.post("/api/v1/simulations/TRUCK-PAUSE/pause").json()["status"] == "PAUSED"
    assert client.post("/api/v1/simulations/TRUCK-PAUSE/stop").json()["status"] == "STOPPED"


def test_tick_completes_route_and_records_final_coordinate(client):
    route = client.post("/api/v1/routes", json=route_request()).json()["routes"][0]
    client.post(
        "/api/v1/simulations",
        json={
            "vehicle_id": "TRUCK-END",
            "route_id": route["route_id"],
            "speed_kmh": 10000,
            "auto_start": False,
        },
    )

    response = client.post(
        "/api/v1/simulations/TRUCK-END/tick",
        json={"advance_seconds": 3600},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "COMPLETED"
    assert response.json()["lat"] == route["geometry"][-1]["lat"]
    assert response.json()["lon"] == route["geometry"][-1]["lon"]
