# Web Scraping (`DataEngineering/WebScraping`)

This directory contains the web scraping and web data extraction modules for the project.

---

## 📁 Modules

### `build_rail_geojson.py`
Extracts Bangkok mass transit lines and station nodes (BTS Sukhumvit, Silom, Gold Line, MRT Blue, Purple, Yellow, Pink Lines) from **OpenStreetMap** via the **Overpass API**, constructing the GeoJSON layer used by the interactive Camera Map.

#### Usage:
```bash
python DataEngineering/WebScraping/build_rail_geojson.py
```

#### Output:
- GeoJSON output saved directly to: `Apps/Camera-Map/static/bangkok_rail.geojson`

---

## 👥 Team Collaboration Note
Additional scraping modules (such as weather/rainfall, incident reports, or city open data) are managed and contributed by the **Web Scraper & BI Dashboard** sub-team into this folder.
