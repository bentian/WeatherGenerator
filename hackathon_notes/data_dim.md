To understand **$D_{\text{loc}}$**, **$D_{\text{glob}}$**, **$Q$**, and **$N_{\text{cells}}$**, consider how WeatherGenerator transforms data:

```
[Raw Observations]
      │  (Patches / Stations / Grids)
      ▼  Stream Embedder
Local Tokens  ──────────►  Shape: (var_tokens, D_loc)
      │  Local Assimilation (Inside each cell)
      ▼  Local→Global Adapter (uses Q queries per cell)
Global Cell Tokens ─────►  Shape: (N_cells, Q, D_glob)
      │  Aggregation & Global Self-Attention
      ▼  Forecast Engine (Rollout)
Global Latent ──────────►  Shape: (N_cells * Q, D_glob)
```

---

### 1. Concrete Config Example

Here is a typical production configuration taken directly from [`config/default_config.yml`](file:///Users/bentian/Workspace/WeatherGenerator/config/default_config.yml):

```yaml
# Global Grid Configuration
healpix_level: 5               # Controls N_cells

# Local (Encoder / Per-Cell) Stage
ae_local_dim_embed: 1024       # D_loc

# Local-to-Global Transition
ae_local_num_queries: 1        # Q

# Global Latent Stage (Encoder backbone, Forecast Engine, Decoder inputs)
ae_global_dim_embed: 2048      # D_glob
```

---

### 2. Parameter Breakdowns with Numerical Examples

#### **$N_{\text{cells}}$ — Number of HEALPix Cells on Earth**
- **Definition:** The Earth's spherical surface is partitioned using the HEALPix hierarchical grid system. The total number of cells covering the globe is given by:
  $$\text{N\_cells} = 12 \times 4^{\text{healpix\_level}}$$
- **Examples:**
  - **`healpix_level: 4`** $\to 12 \times 4^4 = \mathbf{3{,}072}$ cells (~$3.7^\circ$ resolution, ~400 km spacing).
  - **`healpix_level: 5` (Default)** $\to 12 \times 4^5 = \mathbf{12{,}288}$ cells (~$1.8^\circ$ resolution, ~200 km spacing).
  - **`healpix_level: 6`** $\to 12 \times 4^6 = \mathbf{49{,}152}$ cells (~$0.9^\circ$ resolution, ~100 km spacing).
- **Physical context for Taiwan (21.9°–25.3°N, 120°–122°E):**  
  At level 5, each cell is roughly $200\text{ km} \times 200\text{ km}$. Taiwan is only about $390\text{ km} \times 140\text{ km}$, which means the whole island is covered by **only ~2 to 4 cells**!

---

#### **$Q$ (`ae_local_num_queries`) — Latent Tokens per Cell**
- **Definition:** The number of summary tokens representing the atmospheric state inside a single HEALPix cell.
- **Example:**
  - Suppose a single cell over the Taiwan Strait contains:
    - 512 observations from a high-res regional model (WRF/TW_RCM)
    - 64 surface station readings (CWA weather stations)
    - 32 satellite radiance soundings (IASI/SEVIRI)
    - **Total = 608 local tokens** inside that cell.
  - If **`Q = 1`** (default): The cross-attention adapter compresses all 608 tokens into **exactly 1 summary token** of width $D_{\text{glob}}$ for that cell.
  - If **`Q = 4`**: The adapter extracts **4 distinct tokens** per cell (allowing the latent state to separate, for example, vertical layers, land vs. ocean components, or sub-cell variability).
  - **Total latent token count across the globe:**  
    $$N_{\text{cells}} \times Q = 12{,}288 \times 1 = \mathbf{12{,}288}\text{ tokens}$$  
    (or $12{,}288 \times 4 = \mathbf{49{,}152}\text{ tokens}$ if $Q=4$).

---

#### **$D_{\text{loc}}$ (`ae_local_dim_embed`) — Local Embedding Dimension**
- **Definition:** The channel width (vector length) used to represent observations locally before they are compressed into the global grid.
- **Example:**
  - Set to **`1024`** in default config.
  - Every patch of ERA5, satellite pixel, or station measurement is first projected by its stream embedder into a feature vector of length **1024**:
    $$\mathbf{x}_{\text{local}} \in \mathbb{R}^{1024}$$
  - The local self-attention engine ([`LocalAssimilationEngine`](file:///Users/bentian/Workspace/WeatherGenerator/src/weathergen/model/engines.py#L204)) runs varlen multi-head attention among these length-1024 vectors *only within each cell*.

---

#### **$D_{\text{glob}}$ (`ae_global_dim_embed`) — Global Latent Dimension**
- **Definition:** The channel width of the unified global latent space. This is the "backbone" dimension used by:
  1. The global encoder ([`GlobalAssimilationEngine`](file:///Users/bentian/Workspace/WeatherGenerator/src/weathergen/model/engines.py#L456))
  2. The temporal rollout model ([`ForecastingEngine`](file:///Users/bentian/Workspace/WeatherGenerator/src/weathergen/model/engines.py#L544))
  3. The latent memory keys/values queried by the decoders
- **Example:**
  - Set to **`2048`** in default config.
  - The local-to-global adapter projects local tokens ($D_{\text{loc}} = 1024$) to global tokens ($D_{\text{glob}} = 2048$).
  - For each cell, the learnable parameter [`q_cells`](file:///Users/bentian/Workspace/WeatherGenerator/src/weathergen/model/encoder.py#L87-L112) has shape:
    $$(1, Q, D_{\text{glob}}) = (1, 1, 2048)$$
  - During forecast rollout, the state passed from time step $t$ to $t + \Delta t$ is a tensor of shape:
    $$(B, N_{\text{cells}} \cdot Q, D_{\text{glob}}) = (B, 12288, 2048)$$

---

### 3. Summary Tensor Flow

When processing a single global batch ($B=1$) with default parameters:

| Stage | Module | Input Tensor | Output Tensor | Controlled by |
|---|---|---|---|---|
| **Raw Input** | Stream Embedder | Raw channels (e.g., $T, U, V, Q$ patches) | `(N_total_obs, 1024)` | $D_{\text{loc}} = 1024$ |
| **Local Fusion** | `ae_local_engine` | `(N_obs_in_cell, 1024)` | `(N_obs_in_cell, 1024)` | $D_{\text{loc}} = 1024$ |
| **Cell Bottleneck** | `ae_local_global_engine` | `(N_obs_in_cell, 1024)` + `q_cells(1, 2048)` | `(1, 2048)` per cell | $Q = 1, D_{\text{glob}} = 2048$ |
| **Global State** | `ae_global_engine` | `(12288, 2048)` | `(12288, 2048)` | $N_{\text{cells}} = 12288, D_{\text{glob}} = 2048$ |
| **Forecasting** | `forecast_engine` | `(12288, 2048)` at $t$ | `(12288, 2048)` at $t + 6\text{h}$ | $D_{\text{glob}} = 2048$ |
| **Taiwan Decoder** | `TargetPredictionEngine_TW` | Queries at Taiwan coords cross-attending to 9 neighbouring cells ($9 \times Q = 9$ tokens of width $2048$) | Downscaled physical values (e.g. 2m Temperature) | Reads $9 \cdot Q$ tokens of width $D_{\text{glob}}$ |