import React from 'react';
import { Grid, Typography, Paper } from '@mui/material';
import GlobalStatus from './GlobalStatus';
import GlobalControls from './GlobalControls';
import AccountSummary from './AccountSummary';
import PositionsTable from './PositionsTable';
import LogStream from './LogStream';
import InteractiveChart from './InteractiveChart';

const Dashboard = () => {
  return (
    <Grid container spacing={2}>
      {/* Top Bar: Title, Controls, and Status */}
      <Grid item xs={12} md={4}>
        <Typography variant="h4" component="h1" gutterBottom>
          OANDA Live Dashboard
        </Typography>
      </Grid>
      <Grid item xs={12} md={5}>
        <GlobalControls />
      </Grid>
      <Grid item xs={12} md={3}>
        <GlobalStatus />
      </Grid>

      {/* Main Content Area */}
      <Grid item xs={12} lg={8}>
        <Grid container spacing={2}>
          {/* Account Summary */}
          <Grid item xs={12}>
            <AccountSummary />
          </Grid>
          {/* Chart */}
          <Grid item xs={12}>
            <InteractiveChart />
          </Grid>
          {/* Positions Table */}
          <Grid item xs={12}>
            <PositionsTable />
          </Grid>
        </Grid>
      </Grid>

      {/* Right Sidebar: Log Stream */}
      <Grid item xs={12} lg={4}>
        <Paper sx={{ height: '85vh', display: 'flex', flexDirection: 'column' }}>
          <LogStream />
        </Paper>
      </Grid>
    </Grid>
  );
};

export default Dashboard;
