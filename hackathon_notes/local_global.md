In WeatherGenerator’s encoder, **"local"** and **"global"** refer to the **spatial scope of information and attention**:

```
[Raw Observations across different streams: ERA5, Stations, Satellites]
                            │
                            ▼
        ┌──────────────────────────────────────┐
        │                LOCAL                 │
        │  • Inside individual HEALPix cells   │
        │  • Fuses heterogeneous sensors       │
        │  • Isolated (no cross-cell mixing)   │
        └──────────────────┬───────────────────┘
                           │ (Bottleneck: Local→Global Adapter)
                           ▼
        ┌──────────────────────────────────────┐
        │                GLOBAL                │
        │  • Entire planet / all cells         │
        │  • Large-scale circulation & waves   │
        │  • Fixed uniform latent state        │
        └──────────────────────────────────────┘
```

---

### 1. "Local" = Inside a Single HEALPix Cell

- **Spatial scope:** A single HEALPix cell on Earth (at `healpix_level: 5`, this is a patch roughly **~200 km × 200 km**).
- **Engine:** [`LocalAssimilationEngine`](file:///Users/bentian/Workspace/WeatherGenerator/src/weathergen/model/engines.py#L204)
- **Token dimension:** $D_{\text{loc}}$ (`ae_local_dim_embed`, e.g. 512 or 1024).
- **What happens here:**
  - Raw data from different streams (e.g. gridded ERA5 cells, weather station measurements, satellite radiance soundings) are assigned to the HEALPix cell they physically fall into.
  - **Attention stays strictly inside each cell:** tokens only attend to other tokens in the *same* cell. A station in Taiwan will never attend to an observation in Japan at this stage.
- **Why it exists:**  
  1. **Sensor fusion:** It combines heterogeneous observation types (different resolutions, unstructured stations, gridded reanalysis) into a coherent, localized representation of that 200 km patch of atmosphere.
  2. **Computational efficiency:** Processing variable numbers of local points per cell is cheap with block/varlen attention, whereas attending globally across millions of raw observation points worldwide would be impossible ($O(N^2)$ memory explosion).

---

### 2. "Global" = Across the Entire Earth

- **Spatial scope:** The whole globe ($N_{\text{cells}}$ across the entire planetary sphere).
- **Engines:** 
  - [`QueryAggregationEngine`](file:///Users/bentian/Workspace/WeatherGenerator/src/weathergen/model/engines.py#L377) (aggregates across observed cells)
  - [`GlobalAssimilationEngine`](file:///Users/bentian/Workspace/WeatherGenerator/src/weathergen/model/engines.py#L456) (full global self-attention)
- **Token dimension:** $D_{\text{glob}}$ (`ae_global_dim_embed`, e.g. 512 or 2048).
- **What happens here:**
  - Each cell now has a fixed number of summary tokens ($Q$, usually 1).
  - Attention is **planetary-scale**: a cell over Taiwan directly attends to cells over the Pacific Ocean, the Tibetan Plateau, and the Arctic.
- **Why it exists:**  
  Weather is governed by large-scale planetary dynamics (Rossby waves, jet streams, monsoon troughs, subtropical highs). The global stage builds the cohesive planetary state needed by the **forecast engine** to step the atmosphere forward in time.

---

### 3. The Bridge: Local $\to$ Global Adapter

Between them sits the **`ae_local_global_engine`** ([`Local2GlobalAssimilationEngine`](file:///Users/bentian/Workspace/WeatherGenerator/src/weathergen/model/engines.py#L248)):

- Inside each cell, there might be 5 observations or 5,000 observations of width $D_{\text{loc}}$.
- The adapter uses $Q$ learnable queries per cell ([`q_cells`](file:///Users/bentian/Workspace/WeatherGenerator/src/weathergen/model/encoder.py#L87)) to query and compress that variable cloud of local tokens into **exactly $Q$ uniform tokens of width $D_{\text{glob}}$** for every cell on the globe.
- Cells that have zero observations (e.g., masked regions or unobserved oceans) simply retain their prior query embeddings `q_cells`.

---

### Summary Table

| Property | **Local** Stage | **Global** Stage |
|---|---|---|
| **Area covered** | Single cell (~1.8° / ~200 km) | Entire Earth ($360^\circ \times 180^\circ$) |
| **Tokens per cell** | Variable (depends on data density) | Fixed ($Q$, typically 1) |
| **Vector width** | $D_{\text{loc}}$ (`ae_local_dim_embed`) | $D_{\text{glob}}$ (`ae_global_dim_embed`) |
| **Physical purpose** | Local observation fusion (combining sensors, stations, grids) | Planetary atmospheric circulation (Rossby waves, highs/lows) |
| **Modules** | `embed_engine`, `ae_local_engine` | `ae_aggregation_engine`, `ae_global_engine` |