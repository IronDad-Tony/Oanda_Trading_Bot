import { createSlice, PayloadAction } from '@reduxjs/toolkit';

// This interface should match the position data sent from the backend
export interface Position {
  instrument: string;
  long: {
    units: string;
    averagePrice: string;
    pl: string;
  };
  short: {
    units: string;
    averagePrice: string;
    pl: string;
  };
  unrealizedPL: string;
}

interface PositionsState {
  positions: Position[];
  lastUpdated: string | null;
}

const initialState: PositionsState = {
  positions: [],
  lastUpdated: null,
};

export const positionsSlice = createSlice({
  name: 'positions',
  initialState,
  reducers: {
    updatePositions: (state, action: PayloadAction<Position[]>) => {
      state.positions = action.payload;
      state.lastUpdated = new Date().toISOString();
    },
  },
});

export const { updatePositions } = positionsSlice.actions;

export default positionsSlice.reducer;
