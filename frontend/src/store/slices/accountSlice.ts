import { createSlice, PayloadAction } from '@reduxjs/toolkit';

interface AccountState {
  equity: number;
  balance: number;
  marginUsed: number;
  unrealizedPL: number;
  realizedPL: number;
  lastUpdated: string | null;
}

const initialState: AccountState = {
  equity: 0,
  balance: 0,
  marginUsed: 0,
  unrealizedPL: 0,
  realizedPL: 0,
  lastUpdated: null,
};

export const accountSlice = createSlice({
  name: 'account',
  initialState,
  reducers: {
    updateAccount: (state, action: PayloadAction<Omit<AccountState, 'lastUpdated'>>) => {
      state.equity = action.payload.equity;
      state.balance = action.payload.balance;
      state.marginUsed = action.payload.marginUsed;
      state.unrealizedPL = action.payload.unrealizedPL;
      state.realizedPL = action.payload.realizedPL;
      state.lastUpdated = new Date().toISOString();
    },
  },
});

export const { updateAccount } = accountSlice.actions;

export default accountSlice.reducer;
