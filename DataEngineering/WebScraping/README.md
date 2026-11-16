# Geospatial & Web Extraction (`DataEngineering/WebScraping`)

This directory contains the web extraction and geospatial scraping modules designed to harvest external infrastructure datasets that contextualize urban traffic flow in Bangkok.

---

## 📁 Modules

### `build_rail_geojson.py`
Queries the **OpenStreetMap Overpass API** to extract real-world track alignments and station node geometries for all rapid mass transit lines in Greater Bangkok:
- **BTS Sukhumvit Line** (`#65B724`)
- **BTS Silom Line** (`#02817D`)
- **BTS Gold Line** (`#A58704`)
- **MRT Blue Line** (`#1E398D`)
- **MRT Purple Line** (`#893B90`)
- **MRT Yellow Line** (`#FFD400`)
- **MRT Pink Line** (`#E76589`)

#### Operational Workflow:
1. **Route Query**: Downloads transit relations categorized as `subway`, `light_rail`, or `monorail` within the Greater Bangkok bounding box (`13.45°N - 14.05°N, 100.25°E - 100.95°E`).
2. **Station Query & Distance Snapping**: Retrieves railway and public transit station nodes, filters by network operator, and maps each station to its corresponding rail line if within a 150-meter threshold.
3. **Interchange Consolidation**: Merges interchange stations sharing identical names within 300 meters into unified transfer hubs.
4. **GeoJSON Generation**: Outputs an optimized GeoJSON feature collection directly consumable by web mapping engines.

#### Usage:
```bash
python DataEngineering/WebScraping/build_rail_geojson.py
```

#### Output:
- Saved to: `Apps/Camera-Map/static/bangkok_rail.geojson`

---

## 🗺️ Output Schema

The generated `bangkok_rail.geojson` file is formatted as a standard GeoJSON `FeatureCollection` with two feature categories:

### 1. Transit Lines (`properties.kind == "line"`)
- `key`: Unique line identifier (e.g., `bts-sukhumvit`, `mrt-blue`)
- `system`: Mass transit operator (`BTS` or `MRT`)
- `name_th`: Official Thai line name (e.g., `บีทีเอส สายสุขุมวิท`)
- `name_en`: English line name (e.g., `BTS Sukhumvit Line`)
- `colour`: Official hex brand color (e.g., `#65B724`)
- `geometry`: `MultiLineString` containing geographic coordinates `[longitude, latitude]`

### 2. Station Nodes (`properties.kind == "station"`)
- `name_th`: Official Thai station name (e.g., `สยาม`, `อโศก`)
- `name_en`: English station name (e.g., `Siam`, `Asok`)
- `lines`: Array of line keys connected at this station
- `geometry`: `Point` coordinate `[longitude, latitude]`
