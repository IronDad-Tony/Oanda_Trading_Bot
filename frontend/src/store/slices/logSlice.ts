import { createSlice, PayloadAction } from '@reduxjs/toolkit';

export interface LogEntry {
  level: 'INFO' | 'WARN' | 'ERROR' | 'DEBUG' | 'SYSTEM';
  message: string;
  timestamp: string;
}

interface LogState {
  entries: LogEntry[];
}

const MAX_LOG_ENTRIES = 200;

const initialState: LogState = {
  entries: [],
};

export const logSlice = createSlice({
  name: 'logs',
  initialState,
  reducers: {
    addLog: (state, action: PayloadAction<Omit<LogEntry, 'timestamp'>>) => {
      const newEntry: LogEntry = {
        ...action.payload,
        timestamp: new Date().toISOString(),
      };
      // Add to the beginning of the array and keep the array size limited
      state.entries.unshift(newEntry);
      if (state.entries.length > MAX_LOG_ENTRIES) {
        state.entries.pop();
      }
    },
  },
});

export const { addLog } = logSlice.actions;

export default logSlice.reducer;
