# åŒ¯å…¥ PyTorch èˆ‡ Numpy
import torch
import numpy as np
import logging
from typing import Optional, Dict, Any
# åŒ¯å…¥ stable_baselines3 çš„ SAC é¡žåˆ¥ï¼Œç”¨æ–¼æ­£ç¢ºè¼‰å…¥ SAC æ¨¡åž‹
from stable_baselines3 import SAC

logger = logging.getLogger("LiveTradingSystem")

class PredictionService:
    """
    è™•ç†å·²è¨“ç·´æ¨¡åž‹çš„è¼‰å…¥å’Œé æ¸¬ç”Ÿæˆã€‚
    èƒ½å¤ è™•ç†å¤šå€‹äº¤æ˜“å°çš„æ‰¹æ¬¡é æ¸¬ï¼Œä¸¦ç®¡ç† padding/maskingã€‚
    """
    def __init__(self, config: Dict[str, Any]):
        """
        åˆå§‹åŒ– PredictionServiceã€‚
        
        Args:
            config (Dict[str, Any]): ç³»çµ±è¨­å®šæª”ã€‚
        """
        self.model: Optional[object] = None  # å¯èƒ½ç‚º torch.nn.Module æˆ– stable_baselines3.SAC
        self.device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.current_model_path: Optional[str] = None
        self.max_symbols = config.get('max_symbols_allowed', 10) # å¾žè¨­å®šæª”è®€å–
        self.model_lookback_window = config.get('model_lookback_window', 128)
        # æ–°å¢žï¼šè¨˜éŒ„ç›®å‰æ¨¡åž‹æ˜¯å¦ç‚º SAC
        self.is_sac_model: bool = False
        logger.info(f"PredictionService initialized. Max symbols: {self.max_symbols}, Lookback: {self.model_lookback_window}")

    def load_model(self, model_path: str, device: str | None = None):
        """
        å¾žæŒ‡å®šè·¯å¾‘è¼‰å…¥æ¨¡åž‹ï¼Œæ”¯æ´ PyTorch åŠ stable_baselines3 SACã€‚
        è‹¥æª”åæˆ–å‰¯æª”ååŒ…å« 'sac' å­—æ¨£ï¼Œå‰‡ä»¥ stable_baselines3.SAC.load è¼‰å…¥ã€‚
        å…¶é¤˜å‰‡ä»¥ torch.load è¼‰å…¥ã€‚
        """
        if self.current_model_path == model_path and self.model is not None:
            logger.info(f"Model from {model_path} is already loaded.")
            return
        
        # Auto-select GPU when available if device not specified
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        try:
            # åˆ¤æ–·æ˜¯å¦ç‚º SAC æ¨¡åž‹ï¼ˆå¯ä¾å¯¦éš›å‘½åè¦å‰‡èª¿æ•´ï¼‰
            if 'sac' in model_path.lower():
                # ä½¿ç”¨ stable_baselines3 SAC æ–¹å¼è¼‰å…¥
                # è¨­å®š is_sac_model ç‚º True
                self.model = SAC.load(model_path, device=self.device)
                self.is_sac_model = True
                self.current_model_path = model_path
                logger.info(f"SAC æ¨¡åž‹å·²ä½¿ç”¨ stable_baselines3.SAC.load å¾ž {model_path} è¼‰å…¥åˆ° {self.device}ã€‚")
            else:
                # é è¨­ç‚º PyTorch é¡žåž‹æ¨¡åž‹
                self.model = torch.load(model_path, map_location=self.device)
                self.model.to(self.device)
                self.model.eval()
                self.is_sac_model = False
                self.current_model_path = model_path
                logger.info(f"PyTorch æ¨¡åž‹æˆåŠŸå¾ž {model_path} è¼‰å…¥åˆ° {self.device}ã€‚")
        except FileNotFoundError:
            logger.error(f"éŒ¯èª¤: æ¨¡åž‹æª”æ¡ˆä¸å­˜åœ¨æ–¼ {model_path}")
            self.model = None
            self.current_model_path = None
            self.is_sac_model = False
            raise
        except Exception as e:
            logger.error(f"å¾ž {model_path} è¼‰å…¥æ¨¡åž‹æ™‚ç™¼ç”ŸéŒ¯èª¤: {e}", exc_info=True)
            self.model = None
            self.current_model_path = None
            self.is_sac_model = False
            raise

    def predict(self, processed_data_map: Dict[str, np.ndarray]) -> Dict[str, int]:
        """
        æ ¹æ“šä¸€æ‰¹é è™•ç†å¾Œçš„æ•¸æ“šç”Ÿæˆé æ¸¬ã€‚
        è™•ç†å¤šå€‹äº¤æ˜“å°çš„æ•¸æ“šï¼Œé€²è¡Œå¿…è¦çš„å¡«å……å’ŒæŽ©ç¢¼ï¼Œç„¶å¾Œé€²è¡Œæ‰¹æ¬¡é æ¸¬ã€‚
    
        Args:
            processed_data_map (Dict[str, np.ndarray]):
                ä¸€å€‹å­—å…¸ï¼Œéµæ˜¯äº¤æ˜“å°åç¨±ï¼Œå€¼æ˜¯é è™•ç†å¾Œçš„æ•¸æ“š (T, C)ï¼Œ
                å…¶ä¸­ T æ˜¯æ™‚é–“æ­¥é•·ï¼ŒC æ˜¯ç‰¹å¾µæ•¸ã€‚
    
        Returns:
            Dict[str, int]: ä¸€å€‹å­—å…¸ï¼Œéµæ˜¯äº¤æ˜“å°åç¨±ï¼Œå€¼æ˜¯é æ¸¬çš„å‹•ä½œã€‚
            
        Raises:
            ValueError: å¦‚æžœæ¨¡åž‹æœªè¼‰å…¥æˆ–è¼¸å…¥æ•¸æ“šæœ‰å•é¡Œã€‚
        """
        if self.model is None:
            logger.error("æ¨¡åž‹æœªè¼‰å…¥ï¼Œç„¡æ³•é æ¸¬ã€‚")
            raise ValueError("æ¨¡åž‹æœªè¼‰å…¥ã€‚è«‹å…ˆå‘¼å« load_model()ã€‚")
    
        if not processed_data_map:
            logger.warning("æ”¶åˆ°çš„ processed_data_map ç‚ºç©ºï¼Œä¸é€²è¡Œé æ¸¬ã€‚")
            return {}
    
        instruments = list(processed_data_map.keys())
        if len(instruments) > self.max_symbols:
            logger.warning(f"æä¾›çš„äº¤æ˜“å°æ•¸é‡ ({len(instruments)}) è¶…éŽæ¨¡åž‹æœ€å¤§å®¹é‡ ({self.max_symbols})ã€‚å°‡åªä½¿ç”¨å‰ {self.max_symbols} å€‹ã€‚")
            instruments = instruments[:self.max_symbols]
    
        # ç²å–ç‰¹å¾µç¶­åº¦
        first_instrument_data = next(iter(processed_data_map.values()))
        if first_instrument_data.ndim != 2:
            raise ValueError(f"æ•¸æ“šç¶­åº¦ä¸æ­£ç¢ºã€‚é æœŸç‚º 2D (T, C)ï¼Œä½†æ”¶åˆ° {first_instrument_data.ndim}Dã€‚")
        _, num_features = first_instrument_data.shape
    
        # æº–å‚™æ‰¹æ¬¡å¼µé‡å’Œ padding mask
        # Tensor shape: (batch_size, num_symbols, sequence_length, num_features)
        batch_tensor = torch.zeros(1, self.max_symbols, self.model_lookback_window, num_features, dtype=torch.float32)
        # Mask shape: (batch_size, num_symbols) -> True è¡¨ç¤º padded
        padding_mask = torch.ones(1, self.max_symbols, dtype=torch.bool)
    
        active_instruments = []
        for i, instrument in enumerate(instruments):
            data = processed_data_map[instrument]
            
            # è™•ç†é•·åº¦ä¸è¶³çš„æ•¸æ“š (åœ¨æ™‚é–“åºåˆ—é–‹é ­å¡«å……0)
            if data.shape[0] < self.model_lookback_window:
                pad_width = self.model_lookback_window - data.shape[0]
                # åœ¨ç¬¬ä¸€å€‹ç¶­åº¦ (æ™‚é–“) çš„é–‹é ­å¡«å……
                padded_data = np.pad(data, ((pad_width, 0), (0, 0)), 'constant', constant_values=0)
            else:
                # ç¢ºä¿æ•¸æ“šé•·åº¦æ­£ç¢º
                padded_data = data[-self.model_lookback_window:]
    
            batch_tensor[0, i, :, :] = torch.from_numpy(padded_data)
            padding_mask[0, i] = False  # æ¨™è¨˜ç‚ºéžå¡«å……
            active_instruments.append(instrument)
    
        batch_tensor = batch_tensor.to(self.device)
        padding_mask = padding_mask.to(self.device)
    
        try:
            # åˆ¤æ–·ç›®å‰æ¨¡åž‹æ˜¯å¦ç‚º SAC
            if self.is_sac_model:
                # ========== SAC æŽ¨è«–æµç¨‹ ==========
                # é‡å°æ¯å€‹ active_instrumentï¼Œçµ„è£ observation dict ä¸¦é€²è¡ŒæŽ¨è«–
                results = {}
                for i, instrument in enumerate(active_instruments):
                    # å–å¾— observation_space çµæ§‹
                    obs_space = self.model.observation_space

                    # å–å¾— processed_dataï¼Œshape: (lookback_window, num_features)
                    data = processed_data_map[instrument]
                    # è‹¥é•·åº¦ä¸è¶³ï¼Œéœ€èˆ‡å‰é¢ä¸€è‡´é€²è¡Œ padding
                    if data.shape[0] < self.model_lookback_window:
                        pad_width = self.model_lookback_window - data.shape[0]
                        padded_data = np.pad(data, ((pad_width, 0), (0, 0)), 'constant', constant_values=0)
                    else:
                        padded_data = data[-self.model_lookback_window:]

                    # çµ„è£ observation dictï¼Œéœ€åŒ…å« 'context_features', 'market_features', 'symbol_id'
                    # å–å¾—å„ key çš„ shape
                    obs_dict = {}
                    for key in obs_space.spaces:
                        space = obs_space.spaces[key]
                        # ä¾ç…§ key åˆ†é¡žè™•ç†
                        if key == 'market_features':
                            # market_features å¡«å…¥ processed_dataï¼ˆå¦‚æœ‰ context_featuresï¼Œéœ€åˆ†å‰²ï¼‰
                            # è‹¥ processed_data åŒ…å«å¤šæ–¼ä¸€çµ„ç‰¹å¾µï¼Œéœ€æ ¹æ“š shape å°æ‡‰
                            # å–å¾—ç›®æ¨™ shape
                            target_shape = space.shape
                            # è‹¥ processed_data ç‰¹å¾µæ•¸èˆ‡ target_shape ä¸ç¬¦ï¼Œå‰‡è£œ 0
                            if padded_data.shape[-1] >= target_shape[-1]:
                                obs_dict[key] = padded_data[:, :target_shape[-1]].astype(np.float32)
                            else:
                                # ç‰¹å¾µæ•¸ä¸è¶³æ™‚è£œ 0
                                temp = np.zeros((padded_data.shape[0], target_shape[-1]), dtype=np.float32)
                                temp[:, :padded_data.shape[-1]] = padded_data
                                obs_dict[key] = temp
                        elif key == 'context_features':
                            # context_features è‹¥ processed_data æœ‰åŒ…å«ï¼Œå‰‡åˆ†å‰²å‡ºä¾†ï¼Œå¦å‰‡è£œ 0
                            target_shape = space.shape
                            # å‡è¨­ context_features åœ¨ processed_data æœ€å¾Œå¹¾å€‹æ¬„ä½ï¼ˆå¦‚æœ‰ï¼‰ï¼Œå¦å‰‡è£œ 0
                            if padded_data.shape[-1] >= target_shape[-1]:
                                obs_dict[key] = padded_data[:, -target_shape[-1]:].astype(np.float32)
                            else:
                                obs_dict[key] = np.zeros((padded_data.shape[0], target_shape[-1]), dtype=np.float32)
                        elif key == 'symbol_id':
                            # symbol_id ç›´æŽ¥å¡«å…¥ instrument index
                            obs_dict[key] = np.array([i], dtype=np.int32)
                        else:
                            # å…¶ä»– key ä¸€å¾‹è£œ 0
                            if len(space.shape) == 0:
                                obs_dict[key] = np.array(0, dtype=space.dtype if hasattr(space, 'dtype') else np.float32)
                            else:
                                obs_dict[key] = np.zeros(space.shape, dtype=space.dtype if hasattr(space, 'dtype') else np.float32)

                    # ==========================
                    # é€²è¡Œ SAC æ¨¡åž‹æŽ¨è«–
                    # ==========================
                    try:
                        # Align observation to model's observation_space if available
                        obs_space = getattr(self.model, 'observation_space', None)
                        if obs_space is not None and hasattr(obs_space, 'spaces'):
                            # Determine symbols dimension from any known key
                            if 'market_features' in obs_space.spaces:
                                symbols_dim = obs_space.spaces['market_features'].shape[0]
                            elif 'context_features' in obs_space.spaces:
                                symbols_dim = obs_space.spaces['context_features'].shape[0]
                            else:
                                symbols_dim = self.max_symbols

                            fixed_obs = {}

                            # market_features: (S, Fm) -> use last timestep
                            if 'market_features' in obs_space.spaces:
                                _, fm = obs_space.spaces['market_features'].shape
                                mf = np.zeros((symbols_dim, fm), dtype=np.float32)
                                last_row = padded_data[-1].astype(np.float32)
                                if last_row.shape[-1] >= fm:
                                    mf[i, :] = last_row[:fm]
                                else:
                                    mf[i, :last_row.shape[-1]] = last_row
                                fixed_obs['market_features'] = mf

                            # context_features: (S, Fc) -> last timestep, last Fc features
                            if 'context_features' in obs_space.spaces:
                                _, fc = obs_space.spaces['context_features'].shape
                                cf = np.zeros((symbols_dim, fc), dtype=np.float32)
                                last_row = padded_data[-1].astype(np.float32)
                                if last_row.shape[-1] >= fc:
                                    cf[i, :] = last_row[-fc:]
                                else:
                                    cf[i, -last_row.shape[-1]:] = last_row
                                fixed_obs['context_features'] = cf

                            # features_from_dataset: (S, T, Fd) -> recent T steps
                            if 'features_from_dataset' in obs_space.spaces:
                                s, t_steps, fd = obs_space.spaces['features_from_dataset'].shape
                                ffd = np.zeros((symbols_dim, t_steps, fd), dtype=np.float32)
                                window = padded_data[-t_steps:, :]
                                if window.shape[1] >= fd:
                                    window_fd = window[:, :fd].astype(np.float32)
                                else:
                                    tmp = np.zeros((window.shape[0], fd), dtype=np.float32)
                                    tmp[:, :window.shape[1]] = window
                                    window_fd = tmp
                                if window_fd.shape[0] < t_steps:
                                    pad = np.zeros((t_steps - window_fd.shape[0], fd), dtype=np.float32)
                                    window_fd = np.vstack([pad, window_fd])
                                ffd[i, :, :] = window_fd[-t_steps:]
                                fixed_obs['features_from_dataset'] = ffd

                            # padding_mask default zeros (no mask)
                            if 'padding_mask' in obs_space.spaces:
                                fixed_obs['padding_mask'] = np.zeros((symbols_dim,), dtype=np.int32)

                            # symbol_id 0..S-1
                            if 'symbol_id' in obs_space.spaces:
                                fixed_obs['symbol_id'] = np.arange(symbols_dim, dtype=np.int32)

                            action, _ = self.model.predict(fixed_obs, deterministic=True)
                            # Select action for this instrument and discretize to -1/0/1
                            try:
                                action_i = float(action[i])
                            except Exception:
                                action_i = float(action) if np.isscalar(action) else 0.0
                            results[instrument] = int(1 if action_i > 0.05 else (-1 if action_i < -0.05 else 0))
                        else:
                            action, _ = self.model.predict(obs_dict, deterministic=True)
                            ai = action[i] if hasattr(action, '__len__') else action
                            results[instrument] = int(1 if ai > 0.05 else (-1 if ai < -0.05 else 0))
                    except Exception as e:
                        logger.error(f"SAC 推論錯誤: {e}", exc_info=True)
                        results[instrument] = None
                logger.info(f"SAC æ¨¡åž‹æˆåŠŸç”Ÿæˆé æ¸¬: {results}")
                return results
            else:
                # ========== åŽŸæœ¬ PyTorch æ¨¡åž‹æŽ¨è«–æµç¨‹ ==========
                with torch.no_grad():
                    # å‡è¨­æ¨¡åž‹ forward æŽ¥å— (features, padding_mask_symbols)
                    output = self.model(batch_tensor, padding_mask_symbols=padding_mask)
                    
                    # å‡è¨­è¼¸å‡ºæ˜¯åˆ†é¡žçš„ logitsï¼Œå½¢ç‹€ç‚º (batch_size, num_symbols, num_actions)
                    predictions_tensor = torch.argmax(output, dim=2) # (1, max_symbols)
                    
                    results = {}
                    for i, instrument in enumerate(active_instruments):
                        results[instrument] = predictions_tensor[0, i].item()
                    
                    logger.info(f"æˆåŠŸç”Ÿæˆé æ¸¬: {results}")
                    return results
    
        except Exception as e:
            logger.error(f"é æ¸¬æœŸé–“ç™¼ç”ŸéŒ¯èª¤: {e}", exc_info=True)
            # è¿”å›žç©ºå­—å…¸è¡¨ç¤ºå¤±æ•—
            return {}

