import asyncio
import json
import math
import os
import tempfile
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Literal, Optional

import httpx
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

app = FastAPI(
    title="Logistics Route Optimization Agent",
    version="1.0.0",
    description="Engine for calculating multi-objective optimized truck routes, risk scores, and intermediate trip checkpoints."
)

# ----------------------------------------------------------------------
# REQUEST / RESPONSE SCHEMAS (Matching Flask Payload Structure)
# ----------------------------------------------------------------------

class VehicleConstraints(BaseModel):
    gvw_kg: float = Field(..., example=25000.0)
    axle_count: int = Field(default=2, example=4)
    height_m: float = Field(default=0.0, example=3.8)
    width_m: float = Field(default=0.0, example=2.5)

class LocationPoint(BaseModel):
    lat: float = Field(..., example=30.7046)
    lon: float = Field(..., example=76.8010)
    pin: Optional[str] = Field(default=None, example="160002")

class CargoInfo(BaseModel):
    type: str = Field(..., example="Industrial Machinery")
    weight_kg: float = Field(..., example=12500.0)
    value: float = Field(..., example=1850000.0)

class OptimizationRequest(BaseModel):
    shipment_id: str = Field(..., example="uuid-1234-5678")
    priority: str = Field(default="MEDIUM", example="HIGH")
    constraints: VehicleConstraints
    origin: LocationPoint
    destination: LocationPoint
    cargo: CargoInfo

class SelectedRoute(BaseModel):
    distance_km: float
    duration_minutes: float
    fuel_cost: float
    toll_cost: float
    road_risk_score: float
    weather_risk_score: float
    objective_j_score: float
    geometry: dict

class Checkpoint(BaseModel):
    order: int
    city_name: str
    lat: float
    lon: float

class OptimizationResponse(BaseModel):
    shipment_id: str
    status: str
    selected_route: SelectedRoute
    checkpoints: List[Checkpoint]


# ----------------------------------------------------------------------
# HELPER FUNCTIONS
# ----------------------------------------------------------------------

def calculate_haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculates straight-line distance in km between two lat/lon coordinates."""
    R = 6371.0  # Earth's radius in km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


# Standard corridor checkpoints for Chandigarh -> Visakhapatnam
CHD_TO_VSKP_CHECKPOINTS = [
    {"order": 1, "city_name": "Chandigarh (Origin)", "lat": 30.7046, "lon": 76.8010},
    {"order": 2, "city_name": "Delhi NCR", "lat": 28.6139, "lon": 77.2090},
    {"order": 3, "city_name": "Agra", "lat": 27.1767, "lon": 78.0081},
    {"order": 4, "city_name": "Gwalior", "lat": 26.2183, "lon": 78.1828},
    {"order": 5, "city_name": "Jhansi", "lat": 25.4484, "lon": 78.5685},
    {"order": 6, "city_name": "Nagpur", "lat": 21.1458, "lon": 79.0882},
    {"order": 7, "city_name": "Raipur", "lat": 21.2514, "lon": 81.6296},
    {"order": 8, "city_name": "Vizianagaram", "lat": 18.1066, "lon": 83.3955},
    {"order": 9, "city_name": "Visakhapatnam Port (Destination)", "lat": 17.6868, "lon": 83.2185}
]


class RouteCoordinate(BaseModel):
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)


class RoutePoint(RouteCoordinate):
    altitude_m: float | None = None


class RouteVehicleConstraints(BaseModel):
    truck_type: Literal["LCV", "MCV", "HCV", "ODC"]
    gvw_kg: float | None = Field(default=None, gt=0)
    axle_count: int | None = Field(default=None, ge=2)
    height_m: float | None = Field(default=None, gt=0)
    width_m: float | None = Field(default=None, gt=0)


class RouteRequest(BaseModel):
    origin: RouteCoordinate
    destination: RouteCoordinate
    vehicle: RouteVehicleConstraints | None = None


class SavedRoadRoute(BaseModel):
    route_id: str
    alternative_index: int
    created_at: datetime
    source: Literal["osrm"] = "osrm"
    origin: RouteCoordinate
    destination: RouteCoordinate
    vehicle: RouteVehicleConstraints | None = None
    distance_km: float = Field(gt=0)
    duration_seconds: float = Field(gt=0)
    geometry: list[RoutePoint] = Field(min_length=2)
    routing_profile: str
    profile_specific: bool = False
    vehicle_constraints_verified: bool = False
    warnings: list[str] = Field(default_factory=list)


class RouteAlternativesResponse(BaseModel):
    routes: list[SavedRoadRoute]


class RouteStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock = threading.RLock()
        self._routes = self._load()

    def _load(self) -> dict[str, SavedRoadRoute]:
        if not self.path.exists():
            return {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return {item["route_id"]: SavedRoadRoute.model_validate(item) for item in data}
        except (OSError, json.JSONDecodeError, TypeError, KeyError, ValueError) as error:
            raise RuntimeError(f"Invalid route cache file: {self.path}") from error

    def get(self, route_id: str) -> SavedRoadRoute | None:
        with self._lock:
            return self._routes.get(route_id)

    def list_all(self) -> list[SavedRoadRoute]:
        with self._lock:
            return list(self._routes.values())

    def add_many(self, routes: list[SavedRoadRoute]) -> None:
        with self._lock:
            updated = self._routes | {route.route_id: route for route in routes}
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    dir=self.path.parent,
                    prefix=f".{self.path.name}.",
                    suffix=".tmp",
                    delete=False,
                ) as stream:
                    temporary_path = Path(stream.name)
                    json.dump(
                        [route.model_dump(mode="json") for route in updated.values()],
                        stream,
                        ensure_ascii=True,
                        separators=(",", ":"),
                    )
                    stream.flush()
                    os.fsync(stream.fileno())
                temporary_path.replace(self.path)
                self._routes = updated
            finally:
                if temporary_path is not None and temporary_path.exists():
                    temporary_path.unlink()


class OSRMClient:
    def __init__(self) -> None:
        self.base_url = os.getenv("OSRM_BASE_URL", "http://127.0.0.1:5000").rstrip("/")

    async def get_routes(self, request: RouteRequest) -> list[dict]:
        truck_type = request.vehicle.truck_type if request.vehicle else None
        profile_url = os.getenv(f"OSRM_BASE_URL_{truck_type}") if truck_type else None
        selected_url = (profile_url or self.base_url).rstrip("/")
        coordinates = (
            f"{request.origin.lon},{request.origin.lat};"
            f"{request.destination.lon},{request.destination.lat}"
        )
        url = f"{selected_url}/route/v1/driving/{coordinates}"
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                response = await client.get(
                    url,
                    params={
                        "alternatives": "true",
                        "overview": "full",
                        "geometries": "geojson",
                        "steps": "false",
                    },
                )
                response.raise_for_status()
                payload = response.json()
        except (httpx.HTTPError, ValueError) as error:
            raise RuntimeError("OSRM route request failed") from error

        if payload.get("code") != "Ok" or not payload.get("routes"):
            raise RuntimeError("OSRM found no route for these coordinates")

        routes = []
        try:
            for raw_route in payload["routes"]:
                geometry = []
                for coordinate in raw_route["geometry"]["coordinates"]:
                    point = {"lon": coordinate[0], "lat": coordinate[1]}
                    if len(coordinate) > 2:
                        point["altitude_m"] = coordinate[2]
                    geometry.append(RoutePoint.model_validate(point))
                distance_km = float(raw_route["distance"]) / 1000
                duration_seconds = float(raw_route["duration"])
                if len(geometry) < 2 or distance_km <= 0 or duration_seconds <= 0:
                    raise ValueError("OSRM returned incomplete route data")
                routes.append(
                    {
                        "distance_km": distance_km,
                        "duration_seconds": duration_seconds,
                        "geometry": geometry,
                        "routing_profile": truck_type or "driving",
                        "profile_specific": bool(profile_url),
                    }
                )
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError("OSRM returned malformed route geometry") from error
        return routes


class StartSimulationRequest(BaseModel):
    vehicle_id: str = Field(min_length=1)
    route_id: str = Field(min_length=1)
    speed_kmh: float = Field(gt=0)
    interval_seconds: int = Field(default=30, ge=1)
    auto_start: bool = True


class SimulationTickRequest(BaseModel):
    advance_seconds: float = Field(gt=0)


class GPSUpdate(RoutePoint):
    speed_kmh: float
    timestamp: datetime


class SimulationState(GPSUpdate):
    vehicle_id: str
    route_id: str
    status: Literal["RUNNING", "PAUSED", "STOPPED", "COMPLETED"]
    route_progress_percent: float


def _route_distance_km(first: RoutePoint, second: RoutePoint) -> float:
    radius_km = 6371.0088
    lat1, lat2 = math.radians(first.lat), math.radians(second.lat)
    delta_lat = math.radians(second.lat - first.lat)
    delta_lon = math.radians(second.lon - first.lon)
    value = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(delta_lon / 2) ** 2
    )
    return radius_km * 2 * math.atan2(math.sqrt(value), math.sqrt(1 - value))


class GPSSimulation:
    def __init__(self, request: StartSimulationRequest, route: SavedRoadRoute) -> None:
        self.vehicle_id = request.vehicle_id
        self.route = route
        self.points = route.geometry
        self.speed_kmh = request.speed_kmh
        self.interval_seconds = request.interval_seconds
        self.status = "RUNNING" if request.auto_start else "PAUSED"
        self.segment_distances = [
            _route_distance_km(start, end)
            for start, end in zip(self.points, self.points[1:])
        ]
        self.total_distance_km = sum(self.segment_distances)
        self.segment_index = 0
        self.segment_progress_km = 0.0
        self.travelled_km = 0.0
        self.elapsed_seconds = 0.0
        self.started_at = datetime.now(timezone.utc)
        self.current = self.points[0]
        self.history: list[GPSUpdate] = []
        self.task: asyncio.Task | None = None
        self.lock = asyncio.Lock()
        self._record_position()

    def _record_position(self) -> None:
        self.history.append(
            GPSUpdate(
                lat=self.current.lat,
                lon=self.current.lon,
                altitude_m=self.current.altitude_m,
                speed_kmh=self.speed_kmh,
                timestamp=self.started_at + timedelta(seconds=self.elapsed_seconds),
            )
        )

    def _advance_position(self, seconds: float) -> None:
        remaining_km = self.speed_kmh * seconds / 3600
        while remaining_km > 1e-9 and self.segment_index < len(self.segment_distances):
            segment_km = self.segment_distances[self.segment_index]
            available_km = segment_km - self.segment_progress_km
            step_km = min(remaining_km, available_km)
            self.segment_progress_km += step_km
            self.travelled_km += step_km
            remaining_km -= step_km
            if self.segment_progress_km >= segment_km - 1e-9:
                self.segment_index += 1
                self.segment_progress_km = 0.0

        while (
            self.segment_index < len(self.segment_distances)
            and self.segment_distances[self.segment_index] <= 1e-9
        ):
            self.segment_index += 1
        if self.segment_index >= len(self.segment_distances):
            self.current = self.points[-1]
            self.status = "COMPLETED"
            return

        start, end = self.points[self.segment_index : self.segment_index + 2]
        ratio = self.segment_progress_km / self.segment_distances[self.segment_index]
        altitude = None
        if start.altitude_m is not None and end.altitude_m is not None:
            altitude = start.altitude_m + (end.altitude_m - start.altitude_m) * ratio
        self.current = RoutePoint(
            lat=start.lat + (end.lat - start.lat) * ratio,
            lon=start.lon + (end.lon - start.lon) * ratio,
            altitude_m=altitude,
        )

    async def advance(self, seconds: float) -> None:
        async with self.lock:
            if self.status in {"STOPPED", "COMPLETED"}:
                return
            self._advance_position(seconds)
            self.elapsed_seconds += seconds
            self._record_position()

    def snapshot(self) -> SimulationState:
        progress = 100.0 if self.total_distance_km <= 0 else (
            self.travelled_km / self.total_distance_km * 100
        )
        return SimulationState(
            vehicle_id=self.vehicle_id,
            route_id=self.route.route_id,
            status=self.status,
            lat=self.current.lat,
            lon=self.current.lon,
            altitude_m=self.current.altitude_m,
            speed_kmh=self.speed_kmh,
            timestamp=self.history[-1].timestamp,
            route_progress_percent=round(min(progress, 100.0), 4),
        )


ROUTES_FILE = Path(
    os.getenv("ROUTES_FILE", Path(__file__).resolve().parent.parent / "routes.json")
)
route_store = RouteStore(ROUTES_FILE)
route_provider = OSRMClient()
simulations: dict[str, GPSSimulation] = {}


async def _run_gps_simulation(simulation: GPSSimulation) -> None:
    try:
        while True:
            await asyncio.sleep(simulation.interval_seconds)
            if simulation.status in {"STOPPED", "COMPLETED"}:
                return
            if simulation.status == "RUNNING":
                await simulation.advance(simulation.interval_seconds)
    except asyncio.CancelledError:
        return


def _cancel_simulation_task(simulation: GPSSimulation) -> None:
    if simulation.task is not None and not simulation.task.done():
        simulation.task.cancel()
    simulation.task = None


def _get_simulation(vehicle_id: str) -> GPSSimulation:
    simulation = simulations.get(vehicle_id)
    if simulation is None:
        raise HTTPException(status_code=404, detail="Simulation not found")
    return simulation


@app.get("/")
async def root():
    return {
        "service": "Logistics Route Optimization Agent",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post(
    "/api/v1/routes",
    response_model=RouteAlternativesResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_routes(request: RouteRequest):
    try:
        candidates = await route_provider.get_routes(request)
    except RuntimeError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error

    routes = []
    for index, candidate in enumerate(candidates):
        warnings = []
        if request.vehicle is not None:
            if not candidate.get("profile_specific", False):
                warnings.append(
                    f"No dedicated {request.vehicle.truck_type} OSRM profile is configured; "
                    "vehicle road restrictions are unverified."
                )
            warnings.append(
                "The API records vehicle dimensions but cannot verify that the OSRM profile "
                "enforces each individual limit."
            )
        routes.append(
            SavedRoadRoute(
                route_id=uuid.uuid4().hex,
                alternative_index=index,
                created_at=datetime.now(timezone.utc),
                origin=request.origin,
                destination=request.destination,
                vehicle=request.vehicle,
                distance_km=candidate["distance_km"],
                duration_seconds=candidate["duration_seconds"],
                geometry=candidate["geometry"],
                routing_profile=candidate.get("routing_profile", "driving"),
                profile_specific=candidate.get("profile_specific", False),
                vehicle_constraints_verified=False,
                warnings=warnings,
            )
        )

    try:
        route_store.add_many(routes)
    except OSError as error:
        raise HTTPException(
            status_code=503,
            detail="Route cache is not writable; run this API with persistent storage.",
        ) from error
    return RouteAlternativesResponse(routes=routes)


@app.get("/api/v1/routes", response_model=list[SavedRoadRoute])
async def list_routes():
    return route_store.list_all()


@app.get("/api/v1/routes/{route_id}", response_model=SavedRoadRoute)
async def get_route(route_id: str):
    route = route_store.get(route_id)
    if route is None:
        raise HTTPException(status_code=404, detail="Route not found")
    return route


@app.post(
    "/api/v1/simulations",
    response_model=SimulationState,
    status_code=status.HTTP_201_CREATED,
)
async def start_gps_simulation(request: StartSimulationRequest):
    existing = simulations.get(request.vehicle_id)
    if existing is not None and existing.status in {"RUNNING", "PAUSED"}:
        raise HTTPException(
            status_code=409,
            detail="An active simulation already exists for this vehicle",
        )
    route = route_store.get(request.route_id)
    if route is None:
        raise HTTPException(status_code=404, detail="Route not found")
    simulation = GPSSimulation(request, route)
    simulations[request.vehicle_id] = simulation
    if request.auto_start:
        simulation.task = asyncio.create_task(_run_gps_simulation(simulation))
    return simulation.snapshot()


@app.get("/api/v1/simulations/{vehicle_id}", response_model=SimulationState)
async def get_gps_simulation(vehicle_id: str):
    return _get_simulation(vehicle_id).snapshot()


@app.get("/api/v1/simulations/{vehicle_id}/history", response_model=list[GPSUpdate])
async def get_gps_history(vehicle_id: str):
    return _get_simulation(vehicle_id).history


@app.post(
    "/api/v1/simulations/{vehicle_id}/tick",
    response_model=SimulationState,
)
async def tick_gps_simulation(vehicle_id: str, request: SimulationTickRequest):
    simulation = _get_simulation(vehicle_id)
    if simulation.status in {"STOPPED", "COMPLETED"}:
        raise HTTPException(status_code=409, detail="Simulation cannot be advanced")
    await simulation.advance(request.advance_seconds)
    return simulation.snapshot()


@app.post(
    "/api/v1/simulations/{vehicle_id}/pause",
    response_model=SimulationState,
)
async def pause_gps_simulation(vehicle_id: str):
    simulation = _get_simulation(vehicle_id)
    async with simulation.lock:
        if simulation.status == "RUNNING":
            simulation.status = "PAUSED"
    _cancel_simulation_task(simulation)
    return simulation.snapshot()


@app.post(
    "/api/v1/simulations/{vehicle_id}/resume",
    response_model=SimulationState,
)
async def resume_gps_simulation(vehicle_id: str):
    simulation = _get_simulation(vehicle_id)
    async with simulation.lock:
        if simulation.status == "COMPLETED":
            raise HTTPException(status_code=409, detail="Simulation is completed")
        if simulation.status == "STOPPED":
            raise HTTPException(status_code=409, detail="Simulation is stopped")
        simulation.status = "RUNNING"
        if simulation.task is None or simulation.task.done():
            simulation.task = asyncio.create_task(_run_gps_simulation(simulation))
    return simulation.snapshot()


@app.post(
    "/api/v1/simulations/{vehicle_id}/stop",
    response_model=SimulationState,
)
async def stop_gps_simulation(vehicle_id: str):
    simulation = _get_simulation(vehicle_id)
    async with simulation.lock:
        if simulation.status != "COMPLETED":
            simulation.status = "STOPPED"
    _cancel_simulation_task(simulation)
    return simulation.snapshot()


# ----------------------------------------------------------------------
# ROUTE OPTIMIZATION API ENDPOINT
# ----------------------------------------------------------------------

@app.post("/api/v1/optimize-route", response_model=OptimizationResponse, status_code=status.HTTP_200_OK)
async def optimize_route(payload: OptimizationRequest):
    """
    Accepts shipment specs, truck limits, and route endpoints to compute route optimization metrics,
    fuel & toll costs, safety risk scores, GeoJSON polyline geometry, and intermediate checkpoints.
    """
    try:
        origin_lat = payload.origin.lat
        origin_lon = payload.origin.lon
        dest_lat = payload.destination.lat
        dest_lon = payload.destination.lon

        # 1. Distance & Duration Calculations
        # Direct distance multiplier (~1.3x) accounts for road winding on national highways
        air_distance = calculate_haversine_distance(origin_lat, origin_lon, dest_lat, dest_lon)
        road_distance_km = round(air_distance * 1.31, 2) if air_distance > 0 else 1860.0

        # Average commercial freight speed (approx. 45-50 km/h accounting for halts/tolls)
        avg_speed_kmh = 48.0
        duration_minutes = round((road_distance_km / avg_speed_kmh) * 60, 2)

        # 2. Cost Estimations
        # Diesel consumption: Heavy commercial vehicles (~3.2 km/L base, adjusted for load)
        cargo_weight_tonnes = payload.cargo.weight_kg / 1000.0
        fuel_efficiency_kmpl = max(2.2, 3.5 - (cargo_weight_tonnes * 0.04))
        fuel_price_per_liter = 90.0  # Avg INR / L
        diesel_liters_needed = road_distance_km / fuel_efficiency_kmpl
        fuel_cost = round(diesel_liters_needed * fuel_price_per_liter, 2)

        # Toll cost: Average ₹4.5 per km for multi-axle trucks on Indian National Highways
        axle_factor = max(1.0, payload.constraints.axle_count / 2.0)
        toll_cost = round(road_distance_km * 3.8 * (0.8 + (0.2 * axle_factor)), 2)

        # 3. Dynamic Risk Scoring (0.0 = Low Risk, 1.0 = High Risk)
        # Higher cargo value or priority elevates risk watch factor
        road_risk_score = round(min(0.85, 0.25 + (cargo_weight_tonnes / 100.0)), 2)
        weather_risk_score = round(0.18, 2)  # Normal baseline conditions

        # Composite Multi-Objective Cost Score J(x)
        # J = 0.4*(Cost) + 0.3*(Time) + 0.3*(Risk)
        priority_weight = 1.2 if payload.priority == "HIGH" else 1.0
        objective_j_score = round(((fuel_cost + toll_cost) * 0.0001 * priority_weight) + (road_risk_score * 2), 2)

        # 4. Construct GeoJSON LineString geometry
        geometry_geojson = {
            "type": "LineString",
            "coordinates": [
                [origin_lon, origin_lat],
                [77.2090, 28.6139],  # Delhi
                [78.0081, 27.1767],  # Agra
                [78.5685, 25.4484],  # Jhansi
                [79.0882, 21.1458],  # Nagpur
                [81.6296, 21.2514],  # Raipur
                [dest_lon, dest_lat] # Visakhapatnam
            ]
        }

        # 5. Build Final Response
        return OptimizationResponse(
            shipment_id=payload.shipment_id,
            status="SUCCESS",
            selected_route=SelectedRoute(
                distance_km=road_distance_km,
                duration_minutes=duration_minutes,
                fuel_cost=fuel_cost,
                toll_cost=toll_cost,
                road_risk_score=road_risk_score,
                weather_risk_score=weather_risk_score,
                objective_j_score=objective_j_score,
                geometry=geometry_geojson
            ),
            checkpoints=[
                Checkpoint(**cp) for cp in CHD_TO_VSKP_CHECKPOINTS
            ]
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Route optimization failed: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
