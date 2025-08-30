# Transformer + SAC + Strategy Pool Integration Plan

## Objectives

- Build a universal, multi-symbol trading model with fixed observation/action dimensions = `MAX_SYMBOLS_ALLOWED` using padding + masking.
- Ensure ML strategies consume high-dimensional transformer features; classical strategies consume preprocessed raw price/volume sequences.
- Make all strategies modular, standardized “advisors” whose outputs are fused under SAC via a strategy superposition layer (ESS).
- Keep a single model identity per `MAX_SYMBOLS_ALLOWED` value; changing it produces a new universal model.
- Guarantee end-to-end consistency so the trained model loads unchanged in the Live Trading system and operates with OANDA execution constraints (spread, leverage, margin, currency conversion, etc.).

## Current Assessment (Repo)

- Env `trading_env.py` already fixes observation and action dims to `MAX_SYMBOLS_ALLOWED`, returns a Dict with keys:
  - `market_features`: `[num_slots, num_features]` (last timestep per symbol)
  - `context_features`: `[num_slots, 5]`
  - `symbol_id`: `[num_slots]` where `padding_id = num_universe_symbols`
- SAC policy defaults to `EnhancedTransformerFeatureExtractor` (agent/enhanced_feature_extractor.py).
  - Uses a simple `UniversalTransformer` across the symbol dimension (tokens) but does not pass any `src_key_padding_mask` and aggregates using the last token. This causes dummy slots to leak into features and violates padding/masking design.
- Strategy Pool (ESS) exists (`EnhancedStrategySuperposition`, `StrategyPoolManager`) and expects per-asset sequences shaped `[B, N, T, D]` for classical strategies, using transformer features as market state for its attention.
  - In SAC `get_action_dist_params`, ESS currently consumes `obs['market_features']` which is `[B,N,D]` (no time dimension). Needs a consistent shim to support both shapes.
- Constants like `MAX_SYMBOLS_ALLOWED` are duplicated in places (e.g. inside `agent/enhanced_feature_extractor.py`).

## Design Recommendations (Industry Practice Summary)

- Multi-asset RL commonly treats assets as tokens: transformer attends across symbols while masking padded slots; pool outputs via masked mean or attention pooling. Keep seq length constant with padding and track masks.
- Keep observation/action shapes fixed. Use `symbol_id` with an embedding and padding index so dummy slots map to zeros; also provide a key padding mask to all attention layers.
- Feature routing:
  - Transformer input: `[B, N, F]` (symbols as tokens). Optional time dimension can be added later.
  - Transformer output per symbol: `[B, N, D]` → masked mean pooling → `[B, D]` (portfolio-level “super analyst” feature).
  - Strategy pool (classical): consume preprocessed raw price/volume sequences `[B, N, T, F_raw]` (we’ll accept `[B, N, F_raw]` and upcast to `[B,N,1,F_raw]` as a compatibility shim).
  - ESS fuses advisors via learned weights (Gumbel-Softmax/softmax), guided by transformer portfolio features.
- SAC backbone: Multi-input extractor returns a single feature vector `[B, D_total]`. Action space remains `[MAX_SYMBOLS_ALLOWED]`. Mask non-active slots in the env when executing actions.
- Naming and universality: encode `MAX_SYMBOLS_ALLOWED` into model naming. When `MAX_SYMBOLS_ALLOWED` changes, create a new model instance.

## Implementation Plan

1) Transformer Feature Extractor (masking + pooling)
   - Replace last-token aggregation with masked mean across symbols.
   - Compute `src_key_padding_mask = (symbol_id == padding_id)`; set `padding_idx` on symbol embedding.
   - Pass the mask to the transformer encoder. Record activations per-layer for diagnostics as before.
   - Remove local constant duplicates; use `common.config` where needed.

2) SAC + ESS shape handling
   - In `CustomSACPolicy.get_action_dist_params` when ESS is enabled:
     - Accept `obs['market_features']` as `[B,N,D]`, upcast to `[B,N,1,D]` for classical strategies if needed.
     - Keep `market_state_features = features_extractor(obs)` as `[B,D_trans]` to drive ESS attention.
     - Ensure ESS output is `[B,N,1]` → squeeze to `[B,N]` for policy mean.

3) Defaults and configs
   - Set `use_symbol_embedding=True` via `enhanced_model_config.json` (already true).
   - Enable ESS by default in `sac_agent_wrapper` with a minimal `ess_config` sourced from `common.config`/`configs/training/quantum_strategy_config.json`.
   - Remove hardcoded `MAX_SYMBOLS_ALLOWED` from extractors; rely on env observation shapes and `symbol_id` high bound for padding id.

4) Validation Script
   - Add `scripts/validate_pipeline.py` to:
     - Create a small dummy obs dict with 2 active symbols + 8 padded; verify:
       - Transformer returns `[B,D]` consistent with masked mean.
       - ESS path executes with `[B,N,D_raw]` and shims to `[B,N,1,D_raw]`.
       - SAC policy forward works end-to-end for one step.

5) Documentation and cleanup
   - Update README with data-flow diagrams and how padding/masking works.
   - Remove/flag legacy scripts that target `src/...` import paths that are no longer valid if they are not used by CI/tests.

## Task Checklist

- [x] Refactor `agent/enhanced_feature_extractor.py` (masking, pooling, config use)
- [x] Update `agent/sac_policy.py` ESS consumption shapes and masking notes
- [x] Set defaults in `agent/sac_agent_wrapper.py` to enable ESS and pass model config
- [x] Add `scripts/validate_pipeline.py` and run basic checks
- [ ] Unify constants and remove duplicates
- [ ] Update docs and clean obsolete files

## Notes on Live Trading Compatibility

- Live predictor already loads SB3 SAC models and constructs a dict observation; our changes keep the same keys (`market_features`, `context_features`, `symbol_id`), so live inference remains compatible.
- Env continues masking inactive symbols by not mapping slots to symbols; actions for padded slots are no-ops.

## Status

- Draft plan committed. Next: feature extractor refactor (mask + pooling).
