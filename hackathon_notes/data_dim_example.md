To configure a **4-channel, $208 \times 208$ grid** observation dataset in WeatherGenerator, you configure two places:

1. **The Stream Configuration** (e.g., `config/streams/<my_stream>/tw_obs.yml`): defines how the $208 \times 208$ grid is ingested, tokenized, and decoded.
2. **The Model/Training Configuration** (`integration_tests/small_multi_stream.yaml` or your run config): sets the latent dimensions.

---

### 1. Key Mathematical Relationships for $208 \times 208$

- **Total grid points:** $208 \times 208 = \mathbf{43{,}264\text{ points}}$ per timestep.
- **Channels:** 4 variables (e.g., $T, U, V, Q$ or 2t, RH, u10, v10).
- **`token_size` selection:** The tokenizer groups raw observation points into tokens. To avoid padding, `token_size` should ideally divide $43{,}264$ evenly:
  - Common choices: **`64`**, **`128`**, or **`256`**.
    - If `token_size: 64`: $43{,}264 / 64 = \mathbf{676\text{ tokens}}$ per timestep.
    - If `token_size: 128`: $43{,}264 / 128 = \mathbf{338\text{ tokens}}$ per timestep.
    - If `token_size: 256`: $43{,}264 / 256 = \mathbf{169\text{ tokens}}$ per timestep.
  - *Recommendation:* **`128`** or **`256`** provides a good balance between sequence length and local detail for regional grids.

---

### 2. Stream Configuration (`tw_obs.yml`)

Create your stream file in your stream directory:

```yaml
TW_OBS:
  type: anemoi                         # or "mesh" / "obs" depending on your zarr layout
  filenames: ['tw_obs_208x208.zarr']
  stream_id: 101                       # Must be unique among all streams

  # --- Channels (4 channels) ---
  source: ['ch1', 'ch2', 'ch3', 'ch4']
  target: ['ch1', 'ch2', 'ch3', 'ch4']
  geoinfo_channels: ['orography', 'lsm'] # Recommended for regional topography (or [] if none)

  loss_weight: 1.0
  # target_channel_weights: { 'ch1': 2.0, 'ch2': 1.0, 'ch3': 1.0, 'ch4': 1.0 }

  # --- Tokenization ---
  token_size: 128                      # 43,264 / 128 = 338 tokens (evenly divisible)
  tokenize_spacetime: True
  masking_rate: 0.6                    # Ratio of tokens masked during training
  masking_rate_none: 0.05
  max_num_targets: -1                  # -1 decodes all 43,264 points (or e.g. 8192 to subsample in training)

  # --- Input Embedder (Maps [token_size * 4 channels] -> ae_local_dim_embed) ---
  embed:
    net: transformer
    num_tokens: 1                      # Output tokens per input patch
    num_heads: 4                       # Heads for StreamEmbedTransformer
    dim_embed: 256                     # Internal embedding width
    num_blocks: 2

  # --- Target Decoder (Reads latent state back to 208x208 grid) ---
  embed_target_coords:
    net: mlp                           # "mlp" handles complex terrain coordinates better than "linear"
    dim_embed: 256                     # D_dec width

  target_readout:
    num_layers: 2
    num_heads: 4

  pred_head:
    ens_size: 1                        # Set >= 2 (e.g. 8) if using ensemble/probabilistic CRPS loss
    num_layers: 2                      # Final linear layer outputs the 4 channels automatically
```

---

### 3. Model Configuration (`small_multi_stream.yaml`)

In your main config file:

```yaml
# 1. Point to your stream folder
streams_directory: "./config/streams/<my_stream>/"

# 2. Local Dimension (D_loc)
# The StreamEmbedder projects to this dimension. 512 or 1024 is typical.
ae_local_dim_embed: 512                # Matches your small config

# 3. Latent Bottleneck (Q & D_glob)
ae_local_num_queries: 1                # Q = 1 token per HEALPix cell
ae_global_dim_embed: 512               # D_glob = 512

# 4. Regional Overfitting Control
# For a relatively small regional domain like 208x208, prevent overfitting:
pred_dropout_rate: 0.1                 # Dropout inside target token engines
embed_dropout_rate: 0.1                # Dropout inside stream embedder
```

---

### 4. Special Flag: Is this stream Input, Output, or Both?

- **If it is a pure prediction target (e.g., Regional RCM / Station Grid):**  
  Add `diagnostic: True` in `tw_obs.yml`. WeatherGenerator will skip building the input `embed` network and use `Identity`, saving memory.
- **If it is input-only (forcing):**  
  Add `forcing: True` in `tw_obs.yml`. WeatherGenerator will skip building the decoder heads.
- **If both:**  
  Leave both flags unset (as in the snippet above).