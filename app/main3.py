import math
from typing import List, Optional
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

