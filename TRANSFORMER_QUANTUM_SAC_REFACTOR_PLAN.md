# SAC + Transformer + Quantum Strategies Integration Plan

## Goals
- Ensure SAC uses a time-aware Transformer as the feature extractor.
- Route high-dimensional Transformer outputs to ML advisors; route raw preprocessed sequences to traditional advisors.
- Maintain fixed MAX_SYMBOLS_ALLOWED observation/action dimensions via padding + masking.
- Standardize advisor (strategy) interface and ensure correct feature shapes per advisor.
- Keep a universal model per MAX_SYMBOLS_ALLOWED; allow sub-selection via masking.
- Preserve end-to-end compatibility with live trading (OANDA) and realistic training (spread, leverage, margin, FX conversion).

## Architecture Decisions
- Observation dict will include:
  - `features_from_dataset`: float32 [S, T, F] per step (padded to `MAX_SYMBOLS_ALLOWED`).
  - `market_features`: float32 [S, F] last-step snapshot (kept for backward use/UI).
  - `context_features`: float32 [S, C] contextual features (positions, pnl, volatility, margin, etc.).
  - `symbol_id`: int32 [S] with a dedicated `padding_symbol_id` for dummy slots.
  - `padding_mask`: MultiBinary [S] where 1 = active, 0 = dummy (derived from `symbol_id`).
- SAC feature extractor: use `TransformerFeatureExtractor` (time-aware) over `features_from_dataset`.
  - Uses `UniversalTradingTransformer` to encode time per symbol and cross-asset attention.
  - Flattens to [B, S*D] to retain per-symbol information.
- Quantum/advisor layer (ESS):
  - Market-state features: the feature extractor output [B, S*D] (attention input).
  - Strategy inputs: raw sequences `features_from_dataset` [B, S, T, F].
  - All advisors implement `BaseStrategy.forward(asset_features: [B,T,F]) -> [B,1,1]`.
- Padding/masking:
  - Env sets `symbol_id` padding id and `padding_mask`.
  - Transformer masks dummy symbols; outputs zero features for padded slots.
  - Env executes only actions for active symbols; dummy slots ignored.

## Work Plan
1) Env observation expansion
- Add `features_from_dataset` [S,T,F] and `padding_mask` to observation and observation_space.
- Keep `market_features`, `context_features`, `symbol_id` for compatibility.
- Ensure shapes use `MAX_SYMBOLS_ALLOWED` with correct padding.

2) Feature extractor switch
- Replace policy `features_extractor_class` with `TransformerFeatureExtractor`.
- Pass `features_key='features_from_dataset'`, `mask_key='padding_mask'` in kwargs.

3) ESS routing update
- In `CustomSACPolicy.get_action_dist_params`, prefer `features_from_dataset` for `asset_features_batch` (fallback to `market_features`).
- Feed `market_state_features = self.extract_features(obs, self.features_extractor)` into ESS attention.

4) Validation and debugging
- Add `scripts/validate_feature_routing.py` to: build env, sample obs, run extractor, run ESS, validate shapes, run one SAC policy forward.
- Print and assert key shapes: masks, transformer outputs [B,S,D], flattened [B,S*D], ESS outputs [B,S].

5) Documentation and cleanup
- Update `docs/training_data_flow_architecture.md` with the new dataflow.
- Mark deprecated: `features/combined_feature_extractor.py` and outdated `StrategyPoolManager` usage paths; keep code but flag for removal after verification.
- After full validation, remove obsolete scripts/configs identified during tests.

## Acceptance Criteria
- Env observation_space includes `features_from_dataset` and `padding_mask` with correct shapes.
- `TransformerFeatureExtractor` receives [B,S,T,F] and returns [B,S*D]; masks padded slots.
- ESS receives raw sequences [B,S,T,F] and returns actions [B,S].
- With `num_active_symbols=2` and `MAX_SYMBOLS_ALLOWED=10`, transformer weakens dummy features and env ignores masked actions.
- Unit hook script runs end-to-end without dimension errors.

## TODO Checklist
- [x] Env: add sequence + mask to observation_space and _get_observation
- [x] Policy: use `TransformerFeatureExtractor` with `features_from_dataset` + `padding_mask`
- [x] Policy: ESS to use `features_from_dataset` sequences for strategies
- [ ] Validation script added and passing
- [ ] Docs updated; deprecated modules flagged
