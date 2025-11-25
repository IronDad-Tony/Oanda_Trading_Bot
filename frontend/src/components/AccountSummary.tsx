import React from 'react';
import { useSelector } from 'react-redux';
import { RootState } from '../store/store';
import { Paper, Grid, Typography, Box } from '@mui/material';

// Helper component for displaying a single metric
const Metric = ({ label, value, isCurrency = true, colorize = false }: { label: string, value: number, isCurrency?: boolean, colorize?: boolean }) => {
  const formatValue = (val: number) => {
    if (isCurrency) {
      return val.toLocaleString('en-US', { style: 'currency', currency: 'USD' });
    }
    return val.toLocaleString();
  };

  const valueColor = colorize ? (value >= 0 ? 'success.main' : 'error.main') : 'text.primary';

  return (
    <Box>
      <Typography variant="caption" color="text.secondary" sx={{ textTransform: 'uppercase' }}>
        {label}
      </Typography>
      <Typography variant="h6" sx={{ color: valueColor, fontWeight: 'bold' }}>
        {formatValue(value)}
      </Typography>
    </Box>
  );
};

const AccountSummary = () => {
  const accountState = useSelector((state: RootState) => state.account);

  return (
    <Paper sx={{ p: 2 }}>
      <Grid container spacing={2} alignItems="center">
        <Grid item xs={12} sm={6} md={2.5}>
          <Metric label="Equity" value={accountState.equity} />
        </Grid>
        <Grid item xs={12} sm={6} md={2.5}>
          <Metric label="Balance" value={accountState.balance} />
        </Grid>
        <Grid item xs={12} sm={6} md={2.5}>
          <Metric label="Unrealized P/L" value={accountState.unrealizedPL} colorize={true} />
        </Grid>
        <Grid item xs={12} sm={6} md={2.5}>
          <Metric label="Realized P/L" value={accountState.realizedPL} colorize={true} />
        </Grid>
        <Grid item xs={12} md={2} sx={{ textAlign: 'right' }}>
          {accountState.lastUpdated && (
            <Typography variant="caption" color="text.secondary">
              Last Updated: {new Date(accountState.lastUpdated).toLocaleTimeString()}
            </Typography>
          )}
        </Grid>
      </Grid>
    </Paper>
  );
};

export default AccountSummary;
