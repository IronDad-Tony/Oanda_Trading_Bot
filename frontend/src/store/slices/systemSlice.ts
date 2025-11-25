import { createSlice, PayloadAction } from '@reduxjs/toolkit';

export type ConnectionStatus = 'disconnected' | 'connecting' | 'connected' | 'error';

interface SystemState {
  connectionStatus: ConnectionStatus;
  message: string | null;
}

const initialState: SystemState = {
  connectionStatus: 'disconnected',
  message: 'Welcome. Attempting to connect to backend...',
};

export const systemSlice = createSlice({
  name: 'system',
  initialState,
  reducers: {
    setConnectionStatus: (state, action: PayloadAction<ConnectionStatus>) => {
      state.connectionStatus = action.payload;
    },
    setSystemMessage: (state, action: PayloadAction<string | null>) => {
      state.message = action.payload;
    },
  },
});

export const { setConnectionStatus, setSystemMessage } = systemSlice.actions;

export default systemSlice.reducer;
