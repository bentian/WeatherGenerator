In WeatherGenerator, **`anemoi`** and **`obs`** are handled by two fundamentally different readers ([`DataReaderAnemoi`](file:///Users/bentian/Workspace/WeatherGenerator/src/weathergen/datasets/data_reader_anemoi.py#L53) vs. [`DataReaderObs`](file:///Users/bentian/Workspace/WeatherGenerator/src/weathergen/datasets/data_reader_obs.py#L25)) designed for two different data paradigms:

---

### Comparison Summary

| Aspect | `anemoi` | `obs` |
|---|---|---|
| **Primary Use Case** | **Gridded & reanalysis data** (e.g., ERA5, CERRA, regular/unstructured climate grids) | **Sparse point observations** (e.g., weather stations, buoy networks, radiosondes, satellite tracks) |
| **Reader Class** | [`DataReaderAnemoi`](file:///Users/bentian/Workspace/WeatherGenerator/src/weathergen/datasets/data_reader_anemoi.py#L53) (subclasses `DataReaderTimestep`) | [`DataReaderObs`](file:///Users/bentian/Workspace/WeatherGenerator/src/weathergen/datasets/data_reader_obs.py#L25) (subclasses `DataReaderBase`) |
| **Coordinates** | **Fixed static spatial grid** (`ds.latitudes`, `ds.longitudes` constant for all time steps) | **Mobile or sparse irregular points** (coordinates stored dynamically per measurement or time window) |
| **Underlying Library** | Uses ECMWF’s [`anemoi-datasets`](https://github.com/ecmwf/anemoi-datasets) package | Directly opens plain Zarr via Python `zarr` package |
| **Data Array Shape** | Typically 3D: `(time, points, variables)` or `(time, lat, lon, variables)` | 2D table-like array: `(num_measurements, columns)` |
| **Indexing Method** | Continuous datetime index slice (`start` to `end`) | Temporal slice lookups via index arrays (e.g. `idx_<YYYYMMDDHHMM>_1`) |

---

### 1. `anemoi` Zarr Type

- **Intended for:** Global or regional fields where **every time step has the exact same set of spatial coordinates** (e.g., your $208 \times 208$ regular grid, ERA5, regional model outputs).
- **Format details:**
  - Standardized ECMWF Anemoi Zarr format.
  - Contains dataset metadata, coordinate arrays (`latitudes`, `longitudes`), variable lists, and pre-computed normalization statistics (`ds.statistics["mean"]`, `ds.statistics["stdev"]`).
  - Supports ensemble dimensions (`ensemble_member`).
- **Stream config example:**
  ```yaml
  TW_GRID:
    type: anemoi
    filenames: ['tw_rcm_208x208.zarr']
    source: ['2t', '10u', '10v']
    target: ['2t', '10u', '10v']
  ```

---

### 2. `obs` Zarr Type

- **Intended for:** Observation collections where **stations report irregularly, points vary in time, or coordinates are associated with each sample** (e.g., Central Weather Administration weather stations, SYNOP, radiosondes).
- **Format details:**
  - Expects a root Zarr group containing:
    - `data`: 2D Zarr array of shape `(total_records, num_columns)`. Attributes must include `attrs["colnames"]`, `attrs["means"]`, and `attrs["vars"]`.
    - `dates`: Observation datetimes.
    - `idx_<base_date>_1`: Pre-computed hourly index mapping time windows to start/end rows in `data`.
  - Column names must have `"obsvalue"` in observation channels, and specific columns for coordinates (`lat`, `lon` or configured via `coords_channels`).
- **Stream config example:**
  ```yaml
  TW_STATIONS:
    type: obs
    filenames: ['cwa_stations_hourly.zarr']
    base_datetime: "1970-01-01T00:00:00"
    coords_channels: ['lat', 'lon']
    geoinfo_channels: ['elevation']    # Required by obs reader
    source: ['obsvalue_t2m', 'obsvalue_rh']
    target: ['obsvalue_t2m', 'obsvalue_rh']
  ```

---

### Which one should you use for a $208 \times 208$ grid?

If your $208 \times 208$ data is a **fixed regional grid** (every timestep has the same 43,264 grid points):
👉 **Use `type: anemoi`**. It is much easier to create using `anemoi-datasets` / `xarray` and avoids having to construct the table indices and column naming conventions required by the `obs` reader.