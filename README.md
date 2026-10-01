# SupplyNet Testing Server

FastAPI service containing the existing route optimizer plus OSRM-backed route retrieval and fake GPS simulation endpoints.

## Run locally

```powershell
python -m pip install -r requirements.txt
$env:OSRM_BASE_URL = "http://localhost:5000"
uvicorn app.main:app --reload
```

The configured OSRM instance must have the desired road-network extract loaded. For routes across India, use an OSRM deployment built from an India OpenStreetMap extract; this API does not bundle the map data or generate a static list of every road. Set `OSRM_BASE_URL` to that service. To use a separate OSRM instance for a truck profile, set `OSRM_BASE_URL_LCV`, `OSRM_BASE_URL_MCV`, `OSRM_BASE_URL_HCV`, or `OSRM_BASE_URL_ODC` as appropriate.

## Route and GPS API

- `POST /api/v1/routes`: body contains `origin` and `destination` (`lat`, `lon`), with optional `vehicle` (`truck_type`, `gvw_kg`, `axle_count`, `height_m`, `width_m`). Returns and saves OSRM route alternatives.
- `GET /api/v1/routes`: list routes saved by this process.
- `GET /api/v1/routes/{route_id}`: retrieve a saved route and its full geometry.
- `POST /api/v1/simulations`: start a simulation using `vehicle_id`, `route_id`, `speed_kmh`, optional `interval_seconds`, and optional `auto_start`.
- `GET /api/v1/simulations/{vehicle_id}` and `/history`: read the latest GPS position and generated history.
- `POST /api/v1/simulations/{vehicle_id}/tick`, `/pause`, `/resume`, and `/stop`: advance manually or control realtime playback.

Generated routes are stored in `routes.json` by default. Set `ROUTES_FILE` to choose another writable path. The file is a route cache, not an all-India roads dataset. Coordinates, road widths, height/weight limits, and truck access restrictions are only as reliable as the configured OSRM data/profile; this API does not claim to verify vehicle restrictions. Standard OSRM road geometry generally does not include elevation, so `altitude_m` can be absent.

The Vercel configuration in this repository runs serverless functions. Its local filesystem is not durable, and background GPS tasks are not guaranteed to persist between requests. Use a persistent server process and writable storage for saved routes and realtime simulations; manual tick mode is suitable for deterministic API demos.

Interactive API documentation is available at `/docs` while the service is running.
