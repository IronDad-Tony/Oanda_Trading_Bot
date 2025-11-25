import { configureStore } from '@reduxjs/toolkit';
import systemReducer from './slices/systemSlice';
import accountReducer from './slices/accountSlice';
import positionsReducer from './slices/positionsSlice';
import logReducer from './slices/logSlice';

export const store = configureStore({
  reducer: {
    system: systemReducer,
    account: accountReducer,
    positions: positionsReducer,
    logs: logReducer,
  },
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware({
      serializableCheck: false, // Allow non-serializable data for now, e.g., for chart objects if needed
    }),
});

// Infer the `RootState` and `AppDispatch` types from the store itself
export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;
