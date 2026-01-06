SCADA Multitask TCN — Per-File Function Specification

Version: 0.1
Scope: This document specifies responsibilities, public APIs, function signatures, I/O contracts, tensor shapes, and invariants for each file in the proposed repo layout. It is meant to be directly implementable with minimal ambiguity and minimal over-engineering.

======================================================================
A) GLOBAL CONVENTIONS (APPLY EVERYWHERE)
==================================================

A.1 Tensor conventions (PyTorch)

- Time axis is always the middle axis: (B, L, F*) for sequences.
- All model-facing floats are torch.float32.
- Masks are represented as float32 (0.0/1.0) when concatenated to model input; may be stored on disk as uint8/bool.
- Causal assumption: no function may use future timesteps to build inputs at timestep t unless explicitly building targets (Y_true).

A.2 Shapes

- X:        (B, L, F)        engineered standardized features
- M_miss:   (B, L, F)        1 present, 0 missing
- Flags:    (B, L, R)        regime/context flags
- Y_true:   (B, K, F)        future features targets
- y:        (B,) or (B, C)   labels (CE or BCE)
- X_in:     (B, L, F_in)     concatenated: [X_corrupt, M_miss, M_mask, Flags]
- F_in = 3*F + R

A.3 Mask semantics

- M_miss: data availability after resampling + engineering; MUST be explicit.
- M_mask: training mask (Stream A) or all-ones (Stream B / clean inference).
- M_score: scoring mask at inference; passed in the “M_mask slot” to keep architecture constant.
- Constraint: when sampling M_mask or M_score, only sample on elements where M_miss==1 (present values). Missing values are NOT reconstruction targets.

A.4 Placeholder value

- c (fill value for masked elements) is 0.0 in standardized feature space by default.

A.5 Persistence & artifacts

- Artifacts directory structure:
  artifacts/scalers/*.json
  artifacts/registries/*.json
  artifacts/checkpoints/*.pt
  artifacts/calibration/*.json or *.pt
- Every checkpoint MUST include a snapshot of Feature/Flag ordering (registry) and scaler IDs used.

A.6 Logging

- Prefer JSONL structured logs for runs (train/eval/infer).
- Every training step logs: loss components, batch sizes, mask rate, and sanity counts (#present, #masked).

A.7 Minimal dependencies

- pandas for QC/feature/flags construction.
- torch for model/training/inference.
- hydra (or OmegaConf) for config composition.
- numpy for array utilities where helpful.

======================================================================
B) CONFIGS (YAML) — REQUIRED KEYS AND MEANINGS
===============================================

Files under configs/ define the single source of truth for ordering and hyperparams. Code MUST not hardcode feature/flag order.

B.1 configs/config.yaml

- defaults: list of hydra defaults (data/features/flags/model/train/infer)
- experiment_name: str
- seed: int
- device: "cpu"|"cuda"
- output_dir: path

B.2 configs/data/default.yaml
Required keys:

- paths:
  raw_dir: path
  processed_dir: path
  interim_dir: optional path
- sampling:
  dt_minutes: int
- windowing:
  L: int
  K: int
  stride: int
- splits:
  method: "time"|"turbine"|"file"
  train_range / val_range / test_range: (start,end) timestamps if time split
- identifiers:
  turbine_id_col: str
  time_col: str
- labels:
  enabled: bool
  label_source: "file"|"column"|"external"
  label_rule: "window_any"|"window_last"|"horizon_any"
  class_names: list[str] (if enabled)
  multilabel: bool
- qc:
  clamp: dict[channel -> (min,max)]
  winsorize: dict[channel -> (low_q, high_q)] or None
  dedupe_rule: "last"|"mean"|"max" etc

B.3 configs/features/default.yaml
Required keys:

- raw_channels: list[str]
- angular_channels: list[str] subset of raw_channels
- angle_units: "deg"|"rad"
- yaw_error:
  enabled: bool
  wind_dir_col: str
  nacelle_yaw_col: str
  output_name: str
  wrap_range: "[-180,180)"|"[-pi,pi)" (interpreted from units)
- deltas:
  enabled: bool
  channels: list[str] (in engineered space names)
- rolling:
  enabled: bool
  windows: list[int] (timesteps)
  stats: ["mean","std"]
  channels: list[str]
  causal: true (MUST be true)
- fill_value_c: float (default 0.0)
- scaler:
  type: "robust"
  clip_iqr: optional float
- output_ordering:
  explicit: bool
  feature_names: list[str] (if explicit) else derived deterministically

B.4 configs/flags/default.yaml
Required keys:

- flag_names: list[str]
- sources:
  type: "columns"|"external"
  mapping: dict[flag_name -> column_name or rule]
- dtype: "float32" (binary stored 0/1)
- causal_alignment: true

B.5 configs/model/default.yaml
Required keys:

- tcn:
  D: int
  kernel_size: int
  num_blocks: int
  dilations: "powers_of_two"|"explicit"
  dilation_list: optional list[int] if explicit
  dropout: float
  norm: "none"|"weight_norm"|"layer_norm"
- heads:
  recon: {enabled: true}
  forecast: {enabled: true, summary: "last"}
  fault: {enabled: bool, pooling: "mean", C: int, multilabel: bool}
- receptive_field_assert:
  enabled: true
  must_cover_L: true

B.6 configs/train/default.yaml
Required keys:

- batch_size: int
- epochs: int
- p_mask: float
- loss:
  huber_delta: float or "per_feature"
  lambdas: {recon: float, pred: float, fault: float}
  regime_weighting:
  enabled: bool
  weight_map: dict[flag_name -> float] OR single w_regime float
- optimizer:
  name: "adamw"
  lr: float
  weight_decay: float
- scheduler: optional

B.7 configs/infer/default.yaml
Required keys:

- p_score: float
- aggregation:
  name: "masked_mean"|"time_mean_feat_max" etc
  params: dict
- topn_features: int
- calibration:
  enabled: bool
  method: "percentile"|"evt"
  per_turbine: bool
  per_regime: bool
- alerting:
  hysteresis: {T_on: float, T_off: float}
  debounce: {N_on: int, N_off: int}

======================================================================
C) PACKAGE STRUCTURE AND PER-FILE FUNCTION SPECS
================================================

---

src/scada_tcn/__init__.py
---------------------

Purpose:

- Expose package version and key public objects (minimal surface).

Public API:

- __version__: str
- from .contracts import Batch, TrainBatch, InferBatch, ModelOutputs, TrainStepOutputs, InferStepOutputs
- from .registry import FeatureRegistry, FlagRegistry

No heavy imports here (avoid importing torch/pandas at import time).

---

src/scada_tcn/config_schema.py
------------------------------

Purpose:

- Define typed configuration schema and validation helpers.
- Enforce invariants early (RF >= L, F_in formula, consistent C).

Key types:

- dataclasses for:
  - DataConfig, FeatureConfig, FlagConfig, ModelConfig, TrainConfig, InferConfig, RootConfig

Public functions:

1) build_config(cfg: OmegaConf|dict) -> RootConfig

   - Converts Hydra/OmegaConf to typed dataclasses.
   - Performs validation via validate_config.
2) validate_config(cfg: RootConfig, registry: FeatureRegistry|None = None) -> None
   Validations (raise ValueError with actionable message):

   - L > 0, K >= 0, stride > 0
   - If labels enabled: cfg.model.heads.fault.enabled must be true and C matches class_names length.
   - If forecast enabled: K > 0.
   - If recon enabled: p_mask in (0,1), p_score in (0,1) when infer enabled.
   - If receptive_field_assert enabled: RF(model) >= L (requires calling modeling.tcn.receptive_field()).
   - If registry provided: feature_names length == F and flag_names length == R.
   - F_in == 3*F + R whenever F and R are known.
3) resolved_feature_names(cfg: RootConfig) -> list[str]

   - Deterministic derivation of engineered feature names based on feature config.
   - Used by prepare_data to build registry.
4) resolved_flag_names(cfg: RootConfig) -> list[str]

   - Returns cfg.flags.flag_names (already ordered).

Internal helpers (not exported):

- _assert(condition: bool, msg: str) -> None

---

src/scada_tcn/registry.py
-------------------------

Purpose:

- Single source of truth for feature and flag ordering.
- Serialized with checkpoints and preprocessing artifacts to prevent silent drift.

Data structures:

- @dataclass FeatureRegistry:
  - feature_names: list[str]
  - raw_channel_names: list[str]
  - angular_channel_names: list[str]
  - engineered_version: str
- @dataclass FlagRegistry:
  - flag_names: list[str]
  - flags_version: str

Public functions:

1) save_registry(feature_reg: FeatureRegistry, flag_reg: FlagRegistry, path: str) -> None

   - Writes JSON with stable fields and schema version.
2) load_registry(path: str) -> tuple[FeatureRegistry, FlagRegistry]

   - Loads JSON and validates minimal schema.
3) assert_registry_compatible(feature_reg: FeatureRegistry, flag_reg: FlagRegistry, cfg: RootConfig) -> None

   - Checks cfg-derived names match registry exactly, unless cfg explicitly allows overrides.
4) feature_index_map(feature_reg: FeatureRegistry) -> dict[str,int]
5) flag_index_map(flag_reg: FlagRegistry) -> dict[str,int]

---

src/scada_tcn/contracts.py
--------------------------

Purpose:

- Centralize typed batch and output contracts (shapes, optional fields).

Dataclasses (torch tensors unless stated):

1) @dataclass class BatchBase:

   - X: torch.Tensor           (B,L,F) float32
   - M_miss: torch.Tensor      (B,L,F) float32
   - Flags: torch.Tensor       (B,L,R) float32
   - turbine_ids: list[str]|torch.Tensor optional (B,)
   - times: torch.Tensor optional (B,L) or (B,) start time indices
   - meta: dict optional
2) @dataclass class TrainBatch(BatchBase):

   - Y_true: torch.Tensor|None (B,K,F)
   - y: torch.Tensor|None      (B,) or (B,C)
   - has_label: torch.Tensor|None (B,) bool/0-1
3) @dataclass class InferBatch(BatchBase):

   - (no extra mandatory fields)

Outputs:
4) @dataclass class ModelOutputs:

- X_hat: torch.Tensor|None  (B,L,F) for recon path
- Y_hat: torch.Tensor|None  (B,K,F) for forecast path
- p_fault: torch.Tensor|None (B,C)

5) @dataclass class TrainStepOutputs:

   - loss: torch.Tensor scalar
   - losses: dict[str,float] (recon/pred/fault + optional)
   - metrics: dict[str,float] (optional)
   - debug: dict[str,Any] (mask counts etc)
6) @dataclass class InferStepOutputs:

   - outputs: ModelOutputs (from clean stream)
   - s_now: torch.Tensor   (B,) float32
   - feat_contrib: torch.Tensor (B,F) float32
   - top_idx: torch.Tensor (B,topn) int64
   - debug: dict[str,Any]

Utility:

- to_device(batch_or_outputs, device) -> same type
- detach_to_cpu(obj) -> same type with tensors moved to cpu and detached

---

src/scada_tcn/utils/shapes.py
-----------------------------

Purpose:

- Centralized shape and sanity assertions. Used in dataset, train step, infer step, and unit tests.

Public functions:

1) assert_shape(t: torch.Tensor, shape: tuple[int|None,...], name: str) -> None

   - None means “any size”.
   - Raises ValueError with actual vs expected.
2) assert_finite(t: torch.Tensor, name: str) -> None

   - Checks no NaN/Inf.
3) assert_binary_mask(t: torch.Tensor, name: str, tol: float=1e-6) -> None

   - Ensures values are approx in {0,1}.
4) assert_contract_train_batch(batch: TrainBatch, F: int, R: int, K: int|None) -> None
5) assert_contract_infer_batch(batch: InferBatch, F: int, R: int) -> None
6) assert_contract_model_outputs(out: ModelOutputs, B: int, L: int, F: int, K: int, C: int, enabled: dict) -> None

---

src/scada_tcn/utils/seed.py
---------------------------

Purpose:

- Deterministic seeding across random, numpy, torch (and cuda if present).

Public functions:

1) set_global_seed(seed: int, deterministic: bool=True) -> None

   - Sets random.seed, np.random.seed, torch.manual_seed, torch.cuda.manual_seed_all.
   - If deterministic: sets torch.backends.cudnn.deterministic=True and benchmark=False.
2) seed_worker(worker_id: int) -> None

   - For DataLoader workers; uses torch.initial_seed() to set numpy/random.
3) make_torch_generator(seed: int, device: str="cpu") -> torch.Generator

   - Used for reproducible masking sampling independent from global RNG.

---

src/scada_tcn/utils/io.py
-------------------------

Purpose:

- Save/load artifacts: scalers, registries, calibration, checkpoints, reports.

Public functions:

1) ensure_dir(path: str) -> None
2) save_json(obj: dict, path: str) -> None
3) load_json(path: str) -> dict
4) save_torch(obj: dict, path: str) -> None

   - torch.save; handles directory creation.
5) load_torch(path: str, map_location: str|None=None) -> dict
6) atomic_write(path: str, write_fn: Callable[[str],None]) -> None

   - Writes to temp file then rename for robustness.
7) save_parquet(df: pandas.DataFrame, path: str, partition_cols: list[str]|None=None) -> None
8) load_parquet(path: str) -> pandas.DataFrame

Checkpoint format requirements (dictionary):

- "model_state": state_dict
- "optim_state": optional
- "cfg": serialized cfg (OmegaConf.to_container)
- "feature_registry": serialized registry dict
- "flag_registry": serialized registry dict
- "scaler_ref": artifact id/path
- "epoch", "step"

---

src/scada_tcn/utils/logging.py
------------------------------

Purpose:

- Lightweight structured logging utilities (JSONL-friendly) without forcing a framework.

Public functions:

1) get_logger(name: str, output_dir: str|None=None) -> logging.Logger

   - If output_dir provided, writes JSONL file handler.
2) log_kv(logger, step: int, payload: dict) -> None

   - Ensures payload is JSON-serializable; coerces tensors to scalars.
3) flatten_dict(d: dict, prefix: str="") -> dict[str,Any]

   - For metrics/loss logging.
4) format_exception(e: Exception) -> str

======================================================================
D) DATA SUBPACKAGE
==================

---

src/scada_tcn/data/__init__.py
--------------------------

Exports:

- qc_align: run_qc_align
- scalers: RobustScalerParams, fit_robust_scaler, transform_robust
- features: build_features
- flags: build_flags
- windowing: build_window_index, slice_window
- dataset: ScadaWindowDataset, make_dataloaders

---

src/scada_tcn/data/qc_align.py
------------------------------

Purpose:

- Load raw SCADA tables, resample to fixed dt, dedupe, clamp/winsorize, and emit explicit missingness mask at raw level.

Key inputs:

- Raw data assumed in pandas DataFrame or parquet/csv.
- Must handle per-turbine separation.

Public functions:

1) load_raw_tables(raw_dir: str, cfg: DataConfig) -> pandas.DataFrame

   - Returns a single DataFrame with at least [turbine_id_col, time_col] + raw_channels + optional label/flag cols.
   - Minimal cleaning only (parse timestamps, sort).
2) run_qc_align(df_raw: pandas.DataFrame, cfg: DataConfig, feature_cfg: FeatureConfig) -> tuple[pandas.DataFrame, pandas.DataFrame]
   Outputs:

   - df_aligned: indexed by (turbine_id, timestamp) with regular sampling.
     Columns: raw_channels (+ any columns needed for flags/labels if sourced from columns).
     Missing values remain NaN here (do NOT fill yet).
   - df_miss_raw: same index, columns raw_channels, values in {0,1} uint8 (1 present, 0 missing).
     Behavior:
   - Per turbine: resample to dt_minutes using time-based resampling.
   - Deduplicate timestamps per cfg.qc.dedupe_rule before resample.
   - Clamp per-channel min/max if provided.
   - Winsorize per-channel if configured (quantile caps computed on training split only IF a split is known at this stage; otherwise compute globally and log that fact).
   - Output missingness after resampling: present if not NaN after resample.
3) clamp_values(df: pandas.DataFrame, clamp_map: dict[str,tuple[float,float]]) -> pandas.DataFrame
4) winsorize_values(df: pandas.DataFrame, winsor_map: dict[str,tuple[float,float]], group_key: str|None=None) -> pandas.DataFrame

Implementation notes:

- Prefer minimal assumptions: resample rule should be deterministic (e.g., mean within bin).
- Ensure timestamps are timezone-consistent.
- Do not leak future information: QC can use within-bin aggregation, not future bins.

Failure modes:

- Missing turbine_id/time columns -> raise ValueError.
- Empty turbine after filtering -> log and skip.

---

src/scada_tcn/data/scalers.py
-----------------------------

Purpose:

- Robust scaler (median/IQR) fit on training data only; reusable at inference.

Types:

1) @dataclass RobustScalerParams:
   - median: dict[str,float]
   - iqr: dict[str,float]
   - eps: float
   - clip_iqr: float|None
   - feature_names: list[str]
   - fitted_on: dict metadata (date, split rule, etc)

Public functions:

1) fit_robust_scaler(df: pandas.DataFrame, feature_cols: list[str], eps: float=1e-6, clip_iqr: float|None=None) -> RobustScalerParams

   - Computes median and IQR per column using non-missing values only.
2) transform_robust(df: pandas.DataFrame, params: RobustScalerParams) -> pandas.DataFrame

   - For each col: (x - median)/max(iqr, eps).
   - Optionally clip result by clip_iqr (symmetric).
   - Leaves NaNs as NaN (do not fill here).
3) save_scaler(params: RobustScalerParams, path: str) -> None
4) load_scaler(path: str) -> RobustScalerParams

Validation:

- params.feature_names must match columns transformed exactly.

---

src/scada_tcn/data/features.py
------------------------------

Purpose:

- Feature engineering: angle sin/cos, yaw_error, deltas, rolling stats (causal), and construction of engineered missingness mask aligned to engineered features.
- Also fills missing engineered values with fill_value_c AFTER standardization.

Public functions:

1) wrap_angle_diff(a: pandas.Series, b: pandas.Series, units: str, wrap_range: str) -> pandas.Series

   - Computes wrap(a - b) with specified range.
2) encode_angle_sin_cos(angle: pandas.Series, units: str, prefix: str) -> pandas.DataFrame

   - Returns columns: f"{prefix}_sin", f"{prefix}_cos".
   - Keeps NaN where angle is NaN.
3) build_features(
   df_aligned: pandas.DataFrame,
   df_miss_raw: pandas.DataFrame,
   cfg: FeatureConfig,
   scaler: RobustScalerParams|None,
   feature_registry: FeatureRegistry|None,
   fill_missing: bool = True
   ) -> tuple[pandas.DataFrame, pandas.DataFrame, FeatureRegistry]
   Inputs:

   - df_aligned: (turbine_id,timestamp) indexed; raw cols.
   - df_miss_raw: same index; raw missing mask 0/1.
   - scaler: if None, build unscaled engineered features for fitting; otherwise apply scaling.
     Outputs:
   - df_X: engineered standardized features, columns exactly in registry.feature_names order.
   - df_M_miss: engineered missing mask aligned to df_X columns, values 0/1 uint8.
   - feature_registry: produced if None, else validated to match produced columns.
     Behavior details:
   - Step 1: build engineered numeric columns (including sin/cos replacement for angular channels; yaw_error optional).
   - Step 2: derive engineered missingness:
     * sin/cos missing if source angle missing.
     * yaw_error missing if either wind_dir or nacelle_yaw missing.
     * deltas missing if current or previous source missing.
     * rolling mean/std missing if any required window value missing OR if you choose “strict present”; default strict present for correctness.
   - Step 3: scale numeric engineered columns using scaler (robust scaling) where applicable.
     * sin/cos are already bounded; still may scale or leave as-is; default: leave sin/cos unscaled (configurable).
   - Step 4: fill missing engineered values with cfg.fill_value_c (default 0.0) ONLY AFTER df_M_miss computed.
   - Must produce deterministic column order.
   - Must not use future values for deltas/rollings (purely causal).
4) add_deltas(df_X: pandas.DataFrame, df_M: pandas.DataFrame, channels: list[str]) -> tuple[pandas.DataFrame,pandas.DataFrame]
5) add_rolling(df_X: pandas.DataFrame, df_M: pandas.DataFrame, channels: list[str], windows: list[int], stats: list[str]) -> tuple[pandas.DataFrame,pandas.DataFrame]

Failure modes:

- Required columns missing -> ValueError.
- Registry mismatch -> ValueError with diff listing.

---

src/scada_tcn/data/flags.py
---------------------------

Purpose:

- Build regime/context flags aligned to engineered timeline and windowing.
- Flags must be causal and must share the same index as df_X.

Public functions:

1) build_flags(
   df_source: pandas.DataFrame,
   cfg: FlagConfig,
   flag_registry: FlagRegistry|None
   ) -> tuple[pandas.DataFrame, FlagRegistry]
   Inputs:

   - df_source: aligned table with columns needed for flags; index (turbine_id,timestamp).
   - cfg.sources defines mapping rules.
     Outputs:
   - df_Flags: columns exactly in registry.flag_names order, float32 0/1.
   - flag_registry: created if None else validated.
     Behavior:
   - If cfg.sources.type=="columns": each flag is df_source[col].astype(float32).fillna(0)
   - If external: load external flag table and merge on (turbine_id,timestamp); fill missing with 0.
   - Enforce causality: if any rule would require future data, reject (raise).
   - Ensure flags aligned to full index; missing rows -> fill 0.
2) load_external_flags(path: str, cfg: FlagConfig) -> pandas.DataFrame

---

src/scada_tcn/data/windowing.py
-------------------------------

Purpose:

- Convert per-turbine timelines into window indices for training/inference.
- Avoid holding all windows as tensors in memory; store a compact window index table.

Data model:

- df_X, df_M_miss, df_Flags stored per turbine with same index (timestamp) and same length.
- WindowIndex table rows: turbine_id, start_pos (int), end_pos (exclusive), split, has_future (bool), label fields.

Public functions:

1) build_window_index(
   df_X: pandas.DataFrame,
   df_M_miss: pandas.DataFrame,
   df_Flags: pandas.DataFrame,
   cfg: DataConfig,
   labels_df: pandas.DataFrame|None = None
   ) -> pandas.DataFrame
   Behavior:

   - For each turbine timeline length T:
     for start in range(0, T - (L+K) + 1, stride):
     define input slice [start, start+L)
     define target slice [start+L, start+L+K)
   - If forecast disabled: allow K=0; still build windows with has_future=False.
   - Label alignment:
     * window_any: label=1 if any label in [start, start+L) or [start, start+L+K) depending on cfg.labels.label_rule.
     * window_last: label from last timestep in input window.
     * horizon_any: any label in future horizon.
   - Must store any needed label metadata (class id, multilabel vector, has_label).
   - Must store timestamp for start (for traceability).
2) slice_window(
   arr: numpy.ndarray|torch.Tensor,
   start: int,
   length: int
   ) -> same type

   - Utility for slicing preloaded per-turbine arrays.
3) split_window_index(df_index: pandas.DataFrame, cfg: DataConfig) -> dict[str,pandas.DataFrame]

   - Returns {"train":..., "val":..., "test":...} according to split rules.

---

src/scada_tcn/data/dataset.py
-----------------------------

Purpose:

- Torch Dataset/DataLoader that yields EXACT tensors required by the spec.
- Uses processed per-turbine arrays + window index table.

Key design choice (deadline-friendly):

- Preprocessing writes per-turbine tensors to disk (e.g., npz or parquet+to_numpy cached).
- Dataset loads per-turbine arrays lazily with caching.

Types:

1) class ScadaWindowDataset(torch.utils.data.Dataset):
   __init__(
   processed_dir: str,
   window_index: pandas.DataFrame,
   feature_reg: FeatureRegistry,
   flag_reg: FlagRegistry,
   cfg: DataConfig,
   mode: "train"|"val"|"test"|"infer",
   cache_size: int = 8
   )

   - Loads window_index subset for the mode.
   - Maintains LRU cache of per-turbine arrays: X, M_miss, Flags, optional labels timeline.

   __len__() -> int

   __getitem__(idx: int) -> TrainBatch|InferBatch
   Output rules:

   - Always returns X, M_miss, Flags as torch.float32 shaped (L, F)/(L,F)/(L,R) then collate makes (B,...).
   - Train/val/test returns TrainBatch with optional Y_true and label y according to cfg.labels and cfg.windowing.K.
   - Infer returns InferBatch with no Y_true/y required.
2) collate_scada_batches(items: list[TrainBatch|InferBatch]) -> TrainBatch|InferBatch

   - Stacks to (B, ...).
   - Adds turbine_ids, times if present.
   - Ensures dtype float32.
3) make_dataloaders(
   processed_dir: str,
   index_splits: dict[str,pandas.DataFrame],
   feature_reg: FeatureRegistry,
   flag_reg: FlagRegistry,
   cfg: RootConfig
   ) -> dict[str, torch.utils.data.DataLoader]

   - Creates loaders for train/val/test.
   - Uses cfg.train.batch_size for train; smaller for val/test optionally.

On-disk processed format expectation (produced by scripts/prepare_data.py):

- processed_dir/
  turbines/<turbine_id>/
  X.npy or X.parquet (engineered filled)
  M_miss.npy
  Flags.npy
  labels.npy (optional)
  timestamps.npy (optional)
  window_index.parquet
  registries/feature_registry.json
  registries/flag_registry.json
  scalers/robust_scaler.json

Failure modes:

- Feature/flag dims mismatch -> ValueError.
- Window slice out of bounds -> AssertionError (should not happen if index built correctly).

======================================================================
E) MODELING SUBPACKAGE
======================

---

src/scada_tcn/modeling/__init__.py
------------------------------

Exports:

- tcn: TCNBackbone, receptive_field
- heads: ReconHead, ForecastHead, FaultHead
- multitask_tcn: MultiTaskTCN

---

src/scada_tcn/modeling/tcn.py
-----------------------------

Purpose:

- Causal dilated residual TCN backbone producing hidden sequence H: (B,L,D).

Core classes:

1) class CausalConv1d(nn.Module):
   __init__(in_ch: int, out_ch: int, kernel_size: int, dilation: int, bias: bool=True)
   forward(x: torch.Tensor) -> torch.Tensor

   - Input x shape: (B, C_in, L) (channel-first for conv1d).
   - Applies left-padding only so output length remains L.
   - No future leakage.
2) class ResidualBlock(nn.Module):
   __init__(D: int, kernel_size: int, dilation: int, dropout: float, norm: str="none")
   forward(x: torch.Tensor) -> torch.Tensor

   - x: (B, D, L)
   - Two causal conv layers typical; residual connection.
   - Optional normalization (weight_norm on convs OR layer_norm over channels/time depending on choice).
   - Must preserve length and channels.
3) class TCNBackbone(nn.Module):
   __init__(F_in: int, D: int, kernel_size: int, dilations: list[int], dropout: float, norm: str="none")
   forward(x_in: torch.Tensor) -> torch.Tensor

   - x_in: (B, L, F_in) float32
   - Internally transpose to (B, F_in, L)
   - Output H: (B, L, D) (transpose back)
     Additional:
   - store self.dilations, self.kernel_size for RF calc.

Public functions:
4) receptive_field(kernel_size: int, dilations: list[int]) -> int

- Returns RF = 1 + (k-1)*sum(dilations)

5) resolve_dilations(cfg_model: ModelConfig) -> list[int]
   - If powers_of_two: [1,2,4,...] length num_blocks
   - If explicit: validate length == num_blocks

Invariants:

- RF >= L if cfg.model.receptive_field_assert enabled.
- Causality: all padding is left padding only.

---

src/scada_tcn/modeling/heads.py
-------------------------------

Purpose:

- Task heads reading H.

Classes:

1) class ReconHead(nn.Module):
   __init__(D: int, F: int)
   forward(H: torch.Tensor) -> torch.Tensor

   - H: (B,L,D) -> X_hat: (B,L,F)
   - Implement as Linear(D,F) applied timewise or 1x1 conv.
2) class ForecastHead(nn.Module):
   __init__(D: int, K: int, F: int, summary: str="last", hidden: int|None=None)
   forward(H: torch.Tensor) -> torch.Tensor

   - Summary:
     * "last": h = H[:, -1, :]
     * Future extensions allowed but default last.
   - MLP maps D -> K*F then reshape to (B,K,F).
3) class FaultHead(nn.Module):
   __init__(D: int, C: int, pooling: str="mean", multilabel: bool=False, hidden: int|None=None)
   forward(H: torch.Tensor) -> torch.Tensor

   - pooling "mean" default: h = mean over L
   - output:
     * multilabel=False: logits (B,C) for CE
     * multilabel=True: logits (B,C) for BCE

Optional:
4) class RULHead(nn.Module):
   __init__(D:int, hidden:int|None=None)
   forward(H: torch.Tensor) -> torch.Tensor

- returns (B,1)

Notes:

- Heads should return logits, not probabilities (apply softmax/sigmoid outside as needed).

---

src/scada_tcn/modeling/multitask_tcn.py
---------------------------------------

Purpose:

- Wrap backbone + heads; unified forward; minimal branching.

Class:

1) class MultiTaskTCN(nn.Module):
   __init__(F_in: int, F: int, R: int, K: int, C: int, cfg_model: ModelConfig)
   forward(
   X_in: torch.Tensor,
   return_recon: bool=True,
   return_forecast: bool=True,
   return_fault: bool=True
   ) -> ModelOutputs

   - X_in: (B,L,F_in)
   - Returns ModelOutputs with enabled outputs else None.
2) predict_proba_fault(logits: torch.Tensor, multilabel: bool) -> torch.Tensor

   - If multilabel: sigmoid
   - Else: softmax

Invariants:

- Does NOT know about Stream A/B; it only consumes X_in.
- Shape assertions can be optionally enabled in forward with a flag.

======================================================================
F) TRAINING SUBPACKAGE
======================

---

src/scada_tcn/training/__init__.py
------------------------------

Exports:

- masking: sample_mask, apply_mask
- losses: huber_loss, masked_huber_loss, forecast_loss, fault_loss
- step: train_step
- trainer: fit

---

src/scada_tcn/training/masking.py
---------------------------------

Purpose:

- Correct, reproducible mask sampling constrained by M_miss.

Public functions:

1) sample_mask(
   M_miss: torch.Tensor,
   p_mask: float,
   generator: torch.Generator|None=None
   ) -> torch.Tensor

   - M_miss: (B,L,F) float32 0/1
   - Returns M_mask: (B,L,F) float32 0/1 where:
     * If M_miss==0 -> M_mask forced to 1 (do not “hide” missing)
     * If M_miss==1 -> Bernoulli(1-p_mask) keep probability
   - Guarantees: masked positions subset of present positions.
2) apply_mask(
   X: torch.Tensor,
   M_mask: torch.Tensor,
   fill_value: float
   ) -> torch.Tensor

   - Returns X_corrupt = M_mask*X + (1-M_mask)*fill_value
3) build_model_input(
   X_corrupt: torch.Tensor,
   M_miss: torch.Tensor,
   M_mask: torch.Tensor,
   Flags: torch.Tensor
   ) -> torch.Tensor

   - Concatenates to X_in: (B,L, 3F+R) float32
   - Ensures dtype float32.

Debug helpers:
4) mask_stats(M_mask: torch.Tensor, M_miss: torch.Tensor) -> dict[str,float]

- fraction masked among present, etc.

---

src/scada_tcn/training/losses.py
--------------------------------

Purpose:

- Implement loss functions with correct masking/reduction.

Public functions:

1) huber_elementwise(err: torch.Tensor, delta: float) -> torch.Tensor

   - Standard Huber per element.
2) huber_loss(
   y_true: torch.Tensor,
   y_pred: torch.Tensor,
   delta: float,
   reduction: "mean"|"sum"="mean"
   ) -> torch.Tensor
3) masked_huber_loss(
   y_true: torch.Tensor,
   y_pred: torch.Tensor,
   mask: torch.Tensor,
   delta: float,
   eps: float=1e-6
   ) -> torch.Tensor

   - mask: same shape as y_true, 1 means include element in loss.
   - Reduction: sum(loss*mask) / max(sum(mask), eps)
   - This is used for reconstruction where mask = (M_mask==0) AND (M_miss==1).
4) forecast_loss(
   Y_true: torch.Tensor,
   Y_hat: torch.Tensor,
   delta: float
   ) -> torch.Tensor
5) fault_loss(
   logits: torch.Tensor,
   y: torch.Tensor,
   multilabel: bool
   ) -> torch.Tensor

   - multilabel False: CrossEntropyLoss expects y shape (B,) int64
   - multilabel True: BCEWithLogitsLoss expects y shape (B,C) float32
6) apply_regime_weight(
   loss: torch.Tensor,
   Flags: torch.Tensor,
   cfg_train: TrainConfig,
   flag_reg: FlagRegistry
   ) -> torch.Tensor

   - If disabled: returns loss unchanged.
   - If enabled:
     * Compute w_regime per sample (B,) from flags at last timestep or mean over window (choose deterministic rule; default use last timestep flags).
     * Multiply scalar loss by mean(w_regime) OR compute per-sample losses if needed.
   - Minimal implementation: scalar weight derived from OR/any of specified flags.
   - Must log which rule is used.

Notes:

- Keep regime weighting simple: if any “non-normal” flags active => downweight to specified scalar.

---

src/scada_tcn/training/step.py
------------------------------

Purpose:

- Single source of truth for a TRAIN step implementing two-stream training exactly:
  Stream A: masked reconstruction
  Stream B: clean forecast + fault

Public function:

1) train_step(
   model: MultiTaskTCN,
   batch: TrainBatch,
   cfg: RootConfig,
   feature_reg: FeatureRegistry,
   flag_reg: FlagRegistry,
   optimizer: torch.optim.Optimizer,
   generator: torch.Generator|None=None
   ) -> TrainStepOutputs
   Behavior:
   - Assert batch contracts.
   - STREAM A:
     * M_mask_A = sample_mask(batch.M_miss, cfg.train.p_mask, generator)
     * X_corrupt_A = apply_mask(batch.X, M_mask_A, cfg.features.fill_value_c)
     * X_in_A = build_model_input(X_corrupt_A, batch.M_miss, M_mask_A, batch.Flags)
     * out_A = model(X_in_A, return_recon=True, return_forecast=False, return_fault=False)
     * recon_target_mask = (1 - M_mask_A) * batch.M_miss
     * L_rec = masked_huber_loss(batch.X, out_A.X_hat, recon_target_mask, delta=cfg.train.loss.huber_delta)
   - STREAM B:
     * M_mask_B = ones_like(batch.M_miss)
     * X_in_B = build_model_input(batch.X, batch.M_miss, M_mask_B, batch.Flags)
     * out_B = model(X_in_B, return_recon=False, return_forecast=forecast_enabled, return_fault=fault_enabled)
     * L_pred = huber_loss(batch.Y_true, out_B.Y_hat, delta) if forecast enabled else 0
     * L_fault = fault_loss(out_B.p_fault, batch.y, multilabel=cfg.model.heads.fault.multilabel) on labeled subset only:
       - If batch.has_label provided: compute on samples where has_label==1; reduce by count (avoid bias).
   - Optional regime weighting: apply to each loss term if enabled.
   - Total loss: L = λr*L_rec + λp*L_pred + λf*L_fault
   - Backprop: optimizer.zero_grad(); L.backward(); optimizer.step()
   - Return TrainStepOutputs with scalar floats in losses dict and debug mask stats.

Edge handling:

- If forecast enabled but Y_true is None -> raise ValueError.
- If fault enabled but y is None -> treat as no labels and set L_fault=0, log warning once per epoch.

No scheduler stepping here unless cfg requires; keep that in trainer.

---

src/scada_tcn/training/trainer.py
---------------------------------

Purpose:

- Epoch loops, evaluation loops, checkpointing hooks; keep minimal.

Public functions:

1) fit(
   model: MultiTaskTCN,
   loaders: dict[str, DataLoader],   # keys: train, val, (optional) test
   cfg: RootConfig,
   feature_reg: FeatureRegistry,
   flag_reg: FlagRegistry,
   output_dir: str,
   resume_ckpt: str|None=None
   ) -> str
   Behavior:

   - Set seeds.
   - Build optimizer, scheduler (optional).
   - If resume_ckpt: load model/optimizer state and validate registry compatibility.
   - For epoch in 1..epochs:
     * train loop:
       - model.train()
       - for batch: call train_step(...)
       - log step metrics
     * val loop:
       - model.eval() with torch.no_grad()
       - call eval_epoch(...) to compute average losses/metrics
     * checkpoint: save best by val loss; also save last.
   - Returns path to best checkpoint.
2) eval_epoch(
   model: MultiTaskTCN,
   loader: DataLoader,
   cfg: RootConfig,
   feature_reg: FeatureRegistry,
   flag_reg: FlagRegistry
   ) -> dict[str,float]

   - Computes validation losses analogous to training but without optimizer step.
   - For recon: you can reuse Stream A masking with fixed generator seed for determinism OR run recon on clean with a fixed scoring mask; choose one and document (default: reuse Stream A masking with deterministic generator per epoch).
   - Must not leak labels; just compute.
3) build_optimizer(model, cfg_train) -> torch.optim.Optimizer
4) build_scheduler(optimizer, cfg_train) -> optional scheduler

Checkpoint helpers:
5) save_checkpoint(model, optimizer, cfg, feature_reg, flag_reg, scaler_ref, path, epoch, step) -> None

======================================================================
G) INFERENCE SUBPACKAGE
=======================

---

src/scada_tcn/inference/__init__.py
-------------------------------

Exports:

- step: infer_step
- scoring: compute_E_rec, aggregate_score, top_contributors
- alerting: AlertState, update_alert_state

---

src/scada_tcn/inference/scoring.py
----------------------------------

Purpose:

- Compute reconstruction error on masked scoring positions (mask-at-test).
- Aggregate scores and compute per-feature contributions.

Public functions:

1) compute_E_rec(
   X: torch.Tensor,
   X_hat: torch.Tensor,
   M_score: torch.Tensor
   ) -> torch.Tensor

   - Returns E_rec: abs((1-M_score) * (X - X_hat))
   - Shape: (B,L,F)
2) aggregate_score(
   E_rec: torch.Tensor,
   cfg_infer: InferConfig,
   M_score: torch.Tensor|None=None
   ) -> torch.Tensor

   - Returns s_now: (B,)
   - Must implement at least:
     * "masked_mean": sum(E_rec) / max(count_masked, eps)
     * optionally other aggregations.
   - If M_score provided, count_masked = sum(1 - M_score) over (L,F) excluding missing if already constrained.
3) per_feature_contribution(
   E_rec: torch.Tensor,
   M_score: torch.Tensor,
   eps: float=1e-6
   ) -> torch.Tensor

   - Returns contrib: (B,F)
   - Default: for each feature f, mean error over timesteps where masked (1-M_score==1):
     contrib[b,f] = sum_t E_rec[b,t,f] / max(sum_t (1-M_score[b,t,f]), eps)
4) top_contributors(contrib: torch.Tensor, topn: int) -> tuple[torch.Tensor, torch.Tensor]

   - Returns (top_idx, top_vals):
     * top_idx: (B,topn) int64
     * top_vals: (B,topn) float32

Forecast scoring later (optional but specified):
5) compute_E_pred(Y_true: torch.Tensor, Y_hat: torch.Tensor) -> torch.Tensor  # (B,K,F)
6) aggregate_score_later(E_rec, E_pred, cfg_infer) -> torch.Tensor            # (B,)

Notes:

- Keep aggregation choices config-driven and logged.

---

src/scada_tcn/inference/step.py
-------------------------------

Purpose:

- Single source of truth for INFER step:
  Clean stream for outputs + scoring stream for reconstruction score.

Public function:

1) infer_step(
   model: MultiTaskTCN,
   batch: InferBatch,
   cfg: RootConfig,
   feature_reg: FeatureRegistry,
   flag_reg: FlagRegistry,
   generator: torch.Generator|None=None
   ) -> InferStepOutputs
   Behavior:
   - Assert batch contracts.
   - CLEAN OUTPUT STREAM:
     * M_ones = ones_like(M_miss)
     * X_in_I = build_model_input(X, M_miss, M_ones, Flags)
     * out_I = model(X_in_I, return_recon=False, return_forecast=forecast_enabled, return_fault=fault_enabled)
   - SCORING STREAM:
     * M_score = sample_mask(M_miss, p_mask=cfg.infer.p_score, generator)  (same sampler, different p)
     * X_corrupt_S = apply_mask(X, M_score, fill_value_c)
     * X_in_S = build_model_input(X_corrupt_S, M_miss, M_score, Flags)
     * out_S = model(X_in_S, return_recon=True, return_forecast=False, return_fault=False)
     * E_rec = compute_E_rec(X, out_S.X_hat, M_score)
     * s_now = aggregate_score(E_rec, cfg.infer, M_score)
     * contrib = per_feature_contribution(E_rec, M_score)
     * top_idx, top_vals = top_contributors(contrib, cfg.infer.topn_features)
   - Return InferStepOutputs.

Output requirements:

- InferStepOutputs.outputs contains model outputs from clean stream (logits ok).
- s_now and RCA artifacts always produced if recon enabled.

---

src/scada_tcn/inference/alerting.py
-----------------------------------

Purpose:

- Stateful hysteresis + debounce alerting per turbine (and optionally per regime).
- Separate from scoring so it can be unit-tested independently.

Types:

1) @dataclass class AlertState:
   - is_on: bool
   - on_count: int
   - off_count: int
   - last_score: float
   - last_timestamp: optional
   - last_top_features: optional list[str] or indices

Public functions:
2) update_alert_state(
      state: AlertState,
      score: float,
      T_on: float,
      T_off: float,
      N_on: int,
      N_off: int,
      timestamp: Any|None=None,
      top_features: list[int]|None=None
   ) -> AlertState
   Rules:

- If state.is_on is False:
  * if score >= T_on: increment on_count else reset on_count
  * if on_count >= N_on: turn on, reset off_count
- If state.is_on is True:
  * if score <= T_off: increment off_count else reset off_count
  * if off_count >= N_off: turn off, reset on_count
- Store last_score, timestamp, top features for reporting.

3) initial_alert_state() -> AlertState

======================================================================
H) EVALUATION SUBPACKAGE
========================

---

src/scada_tcn/evaluation/__init__.py
--------------------------------

Exports:

- metrics: compute_forecast_metrics, compute_fault_metrics, compute_anomaly_metrics
- reports: write_summary_reports

---

src/scada_tcn/evaluation/metrics.py
-----------------------------------

Purpose:

- Compute offline metrics for judge/ops: forecast error, fault classification, anomaly detection curves.

Public functions:

1) compute_forecast_metrics(
   Y_true: numpy.ndarray|torch.Tensor,
   Y_hat: numpy.ndarray|torch.Tensor,
   mask: numpy.ndarray|torch.Tensor|None=None
   ) -> dict[str,float]

   - Metrics: MAE, RMSE overall; optionally per-feature means.
   - If mask provided: apply mask to ignore missing targets.
2) compute_fault_metrics(
   y_true: numpy.ndarray|torch.Tensor,
   logits: numpy.ndarray|torch.Tensor,
   multilabel: bool,
   class_names: list[str]|None=None
   ) -> dict[str,float]

   - For multiclass: accuracy, macro F1, AUROC if feasible (one-vs-rest).
   - For multilabel: micro/macro F1, AUROC per class if feasible.
3) compute_anomaly_metrics(
   scores: numpy.ndarray,
   y_anom: numpy.ndarray,
   thresholds: list[float]|None=None
   ) -> dict[str,float]

   - PR-AUC, ROC-AUC, best F1 threshold on validation.
   - Assumes y_anom is 0/1 labels (derived rule).

Notes:

- Keep evaluation robust to partial labels; return NaN or omit metrics if not computable.

---

src/scada_tcn/evaluation/reports.py
-----------------------------------

Purpose:

- Generate competition/ops friendly reports (JSON/CSV), including RCA examples.

Public functions:

1) write_summary_reports(
   output_dir: str,
   cfg: RootConfig,
   metrics: dict[str,Any],
   examples: dict[str,Any]|None=None
   ) -> None

   - Writes:
     * metrics.json
     * metrics_flat.csv
     * rca_examples.json (optional)
   - Must be deterministic ordering for diffs.
2) build_rca_showcase(
   turbine_ids: list[str],
   timestamps: list[Any],
   scores: list[float],
   top_idx: numpy.ndarray,
   feature_reg: FeatureRegistry,
   topn: int
   ) -> list[dict]

   - Produces human-readable feature names for top contributors.

======================================================================
I) SCRIPTS (CLI ENTRYPOINTS)
============================

Each script is a thin wrapper: parse config via Hydra, call src/ code, write artifacts.

---

scripts/prepare_data.py
-----------------------

Purpose:

- End-to-end preprocessing:
  raw -> qc_align -> feature engineering -> scaler fit/transform -> flags -> write processed arrays -> build window index.

Main function:

1) main(cfg: RootConfig) -> None
   Pipeline (must be followed in this order):

- Load raw: df_raw = load_raw_tables(cfg.data.paths.raw_dir, cfg.data)
- QC/align: df_aligned, df_miss_raw = run_qc_align(df_raw, cfg.data, cfg.features)
- Build preliminary engineered features WITHOUT scaling (or scale with placeholder) to get feature names:
  df_X_unscaled, df_M_miss, feature_reg = build_features(..., scaler=None, feature_registry=None)
- Fit scaler on training split only:
  - Determine train timestamps/turbines via cfg.data.splits
  - Fit robust scaler on df_X_unscaled[train_rows] for numeric subset
- Transform to scaled df_X: transform_robust then reinsert sin/cos columns if leaving unscaled
- Fill missing with c; ensure df_M_miss correct
- Flags: df_Flags, flag_reg = build_flags(df_aligned or merged source, cfg.flags, None)
- Persist artifacts:
  save_scaler(...)
  save_registry(...)
- Persist processed per turbine:
  - Write arrays X, M_miss, Flags (+ labels timeline if enabled) to processed_dir/turbines/`<id>`/
- Build window index: df_index = build_window_index(...) and split it; save processed_dir/window_index.parquet.

Operational requirements:

- Log counts: turbines, rows, missing fraction per feature, and final F/R.

---

scripts/train.py
----------------

Purpose:

- Train model using two-stream loss; save checkpoint(s).

Main function:

1) main(cfg: RootConfig) -> None
   Steps:

- Load registries + scaler references from processed_dir (validate vs cfg).
- Load window_index splits; create dataloaders.
- Instantiate model with computed dims F,R,K,C and cfg.model.
- Call fit(...) and write best checkpoint path to a text file.

Outputs:

- artifacts/checkpoints/best.pt, last.pt
- artifacts/logs/train.jsonl
- artifacts/registries snapshot copied into run directory

---

scripts/infer.py
----------------

Purpose:

- Run inference on dataset or streaming simulation; output scores, top contributors, and (optional) alert states.

Main function:

1) main(cfg: RootConfig) -> None
   Steps:

- Load checkpoint; load registry snapshot and validate.
- Build infer DataLoader (likely test split or a user-specified range).
- For each batch:
  out = infer_step(...)
  collect:
  - s_now
  - top_idx -> feature names
  - p_fault logits/probas
  - Y_hat
- If calibration enabled: apply thresholds (loaded or fit on baseline) before alerting.
- If alerting enabled: maintain per-turbine AlertState and emit events.

Outputs:

- artifacts/infer/results.parquet or csv:
  turbine_id, start_time, s_now, top_features, p_fault (optional), etc
- artifacts/infer/alerts.jsonl (if alerting)

---

scripts/evaluate.py
-------------------

Purpose:

- Offline evaluation and reporting.

Main function:

1) main(cfg: RootConfig) -> None
   Steps:

- Load inference outputs (or recompute).
- If Y_true available: compute forecast metrics.
- If fault labels available: compute fault metrics.
- If anomaly labels derived/available: compute anomaly metrics.
- Write summary reports via write_summary_reports.

Outputs:

- artifacts/reports/metrics.json, metrics.csv, rca_examples.json

======================================================================
J) TOP-LEVEL REPO FILES (NON-CODE) — REQUIRED CONTENT
======================================================

pyproject.toml

- Must define:
  - project name scada_tcn
  - src-layout
  - dependencies (torch, pandas, hydra-core, numpy, pyarrow recommended)
  - tooling: black/ruff/pytest minimal

requirements.txt

- Minimal pinned deps for deadline stability (pin torch, pandas, hydra-core, numpy, pyarrow).

.gitignore

- Must ignore: data/raw, data/processed, artifacts, __pycache__, .pytest_cache, wandb if used.

LICENSE

- Clear license selection; if private, still include for competition clarity.

======================================================================
K) IMPLEMENTATION PRIORITIES (TO KEEP THIS BUILDABLE IN ~1.5 MONTHS)
====================================================================

Priority 1 (Week 1–2): Contracts + Dataset + Model forward

- Implement contracts.py, shapes.py, masking.py, tcn.py, heads.py, multitask_tcn.py, dataset.py.
- Add a “smoke test” script that constructs synthetic data and runs train_step + infer_step.

Priority 2 (Week 2–3): Preprocessing pipeline

- Implement qc_align.py, scalers.py, features.py, flags.py, windowing.py, prepare_data.py.
- Validate with a small sample dataset end-to-end.

Priority 3 (Week 3–5): Training + checkpointing + inference output formats

- trainer.py + train.py + infer.py + scoring.py.

Priority 4 (Week 5–6): Calibration/alerting + evaluation + reports

- alerting.py + evaluate.py + metrics.py + reports.py.
- Keep calibration simple (percentile per turbine/regime) before attempting EVT.

======================================================================
L) NON-NEGOTIABLE INVARIANTS (ASSERT THESE EARLY AND OFTEN)
===========================================================

- Feature and flag ordering MUST be identical between preprocessing, training, and inference:
  registry.feature_names and registry.flag_names are the authority.
- F_in MUST equal 3*F + R; X_in concatenation order fixed:
  [X_corrupt, M_miss, M_mask, Flags]
- All convolutions are causal (left padding only).
- Reconstruction loss is computed ONLY on masked-and-present elements:
  recon_target_mask = (1 - M_mask) * M_miss
- Inference reconstruction score uses mask-at-test with M_score in the M_mask slot.

END OF SPEC
