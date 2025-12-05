/**
 * Professional Trading Dashboard - 專業級交易終端
 * 統一管理訓練系統、實盤交易、模型倉庫
 * 
 * Features:
 * - Real-time WebSocket updates
 * - TradingView-style charts
 * - Responsive design
 * - Dark mode optimized
 */

import React, { useState, useEffect, useCallback, useMemo } from 'react';
import {
  Box,
  Grid,
  Paper,
  Typography,
  Tabs,
  Tab,
  IconButton,
  Chip,
  LinearProgress,
  Button,
  Switch,
  FormControlLabel,
  Divider,
  Card,
  CardContent,
  Tooltip,
  Badge,
  Alert,
  Snackbar,
} from '@mui/material';
import {
  PlayArrow,
  Stop,
  Pause,
  Refresh,
  Settings,
  Notifications,
  ShowChart,
  AccountBalance,
  Speed,
  Memory,
  Storage,
  TrendingUp,
  TrendingDown,
  Circle,
} from '@mui/icons-material';

// Types
interface SystemStatus {
  training: 'idle' | 'running' | 'paused' | 'error';
  liveTrading: 'stopped' | 'running' | 'paused' | 'error';
  api: 'connected' | 'disconnected';
  gpu: {
    available: boolean;
    usage: number;
    memory: number;
    temperature: number;
  };
}

interface AccountInfo {
  balance: number;
  equity: number;
  unrealizedPL: number;
  realizedPL: number;
  marginUsed: number;
  marginAvailable: number;
  currency: string;
}

interface Position {
  instrument: string;
  units: number;
  unrealizedPL: number;
  averagePrice: number;
  currentPrice: number;
}

interface TrainingMetrics {
  step: number;
  totalSteps: number;
  episode: number;
  reward: number;
  loss: number;
  winRate: number;
  eta: string;
}

// API Configuration
const API_BASE = 'http://localhost:8000';
const WS_URL = 'ws://localhost:8000/ws';

// Status indicator component
const StatusIndicator = ({ status }: { status: string }) => {
  const colorMap: Record<string, string> = {
    running: '#4caf50',
    connected: '#4caf50',
    idle: '#9e9e9e',
    stopped: '#9e9e9e',
    paused: '#ff9800',
    error: '#f44336',
    disconnected: '#f44336',
  };

  return (
    <Circle sx={{ fontSize: 12, color: colorMap[status] || '#9e9e9e', mr: 1 }} />
  );
};

// Metric Card Component
const MetricCard = ({
  title,
  value,
  subtitle,
  icon,
  trend,
  color = 'primary',
}: {
  title: string;
  value: string | number;
  subtitle?: string;
  icon?: React.ReactNode;
  trend?: 'up' | 'down' | 'neutral';
  color?: 'primary' | 'success' | 'error' | 'warning';
}) => {
  const trendColor = trend === 'up' ? '#4caf50' : trend === 'down' ? '#f44336' : '#9e9e9e';

  return (
    <Card sx={{ height: '100%', bgcolor: 'background.paper' }}>
      <CardContent>
        <Box display="flex" justifyContent="space-between" alignItems="flex-start">
          <Box>
            <Typography variant="caption" color="text.secondary" sx={{ textTransform: 'uppercase' }}>
              {title}
            </Typography>
            <Typography variant="h5" sx={{ fontWeight: 'bold', color: `${color}.main` }}>
              {value}
            </Typography>
            {subtitle && (
              <Typography variant="caption" color="text.secondary">
                {subtitle}
              </Typography>
            )}
          </Box>
          {icon && (
            <Box sx={{ color: 'text.secondary' }}>
              {icon}
            </Box>
          )}
        </Box>
        {trend && (
          <Box display="flex" alignItems="center" mt={1}>
            {trend === 'up' ? <TrendingUp sx={{ color: trendColor, fontSize: 16 }} /> :
             trend === 'down' ? <TrendingDown sx={{ color: trendColor, fontSize: 16 }} /> : null}
          </Box>
        )}
      </CardContent>
    </Card>
  );
};

// System Status Panel
const SystemStatusPanel = ({ status }: { status: SystemStatus }) => {
  return (
    <Paper sx={{ p: 2 }}>
      <Typography variant="h6" gutterBottom>
        System Status
      </Typography>
      <Grid container spacing={2}>
        <Grid item xs={6}>
          <Box display="flex" alignItems="center">
            <StatusIndicator status={status.training} />
            <Typography variant="body2">
              Training: {status.training}
            </Typography>
          </Box>
        </Grid>
        <Grid item xs={6}>
          <Box display="flex" alignItems="center">
            <StatusIndicator status={status.liveTrading} />
            <Typography variant="body2">
              Live: {status.liveTrading}
            </Typography>
          </Box>
        </Grid>
        <Grid item xs={6}>
          <Box display="flex" alignItems="center">
            <StatusIndicator status={status.api} />
            <Typography variant="body2">
              API: {status.api}
            </Typography>
          </Box>
        </Grid>
        <Grid item xs={6}>
          {status.gpu.available && (
            <Box display="flex" alignItems="center">
              <Memory sx={{ fontSize: 16, mr: 1, color: 'primary.main' }} />
              <Typography variant="body2">
                GPU: {status.gpu.usage}%
              </Typography>
            </Box>
          )}
        </Grid>
      </Grid>
    </Paper>
  );
};

// Training Panel Component
const TrainingPanel = ({
  metrics,
  onStart,
  onStop,
  onPause,
  isRunning,
}: {
  metrics: TrainingMetrics;
  onStart: () => void;
  onStop: () => void;
  onPause: () => void;
  isRunning: boolean;
}) => {
  const progress = metrics.totalSteps > 0 ? (metrics.step / metrics.totalSteps) * 100 : 0;

  return (
    <Paper sx={{ p: 2, height: '100%' }}>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
        <Typography variant="h6">Training Center</Typography>
        <Box>
          <IconButton onClick={onStart} disabled={isRunning} color="success">
            <PlayArrow />
          </IconButton>
          <IconButton onClick={onPause} disabled={!isRunning} color="warning">
            <Pause />
          </IconButton>
          <IconButton onClick={onStop} disabled={!isRunning} color="error">
            <Stop />
          </IconButton>
        </Box>
      </Box>

      <Box mb={2}>
        <Box display="flex" justifyContent="space-between" mb={1}>
          <Typography variant="body2">Progress</Typography>
          <Typography variant="body2">{progress.toFixed(1)}%</Typography>
        </Box>
        <LinearProgress
          variant="determinate"
          value={progress}
          sx={{ height: 8, borderRadius: 4 }}
        />
        <Typography variant="caption" color="text.secondary">
          Step {metrics.step.toLocaleString()} / {metrics.totalSteps.toLocaleString()} | ETA: {metrics.eta}
        </Typography>
      </Box>

      <Grid container spacing={2}>
        <Grid item xs={6}>
          <Typography variant="caption" color="text.secondary">Episode</Typography>
          <Typography variant="h6">{metrics.episode}</Typography>
        </Grid>
        <Grid item xs={6}>
          <Typography variant="caption" color="text.secondary">Win Rate</Typography>
          <Typography variant="h6" color={metrics.winRate >= 0.5 ? 'success.main' : 'error.main'}>
            {(metrics.winRate * 100).toFixed(1)}%
          </Typography>
        </Grid>
        <Grid item xs={6}>
          <Typography variant="caption" color="text.secondary">Reward</Typography>
          <Typography variant="h6">{metrics.reward.toFixed(4)}</Typography>
        </Grid>
        <Grid item xs={6}>
          <Typography variant="caption" color="text.secondary">Loss</Typography>
          <Typography variant="h6">{metrics.loss.toFixed(4)}</Typography>
        </Grid>
      </Grid>
    </Paper>
  );
};

// Positions Table Component
const PositionsPanel = ({ positions }: { positions: Position[] }) => {
  return (
    <Paper sx={{ p: 2 }}>
      <Typography variant="h6" gutterBottom>
        Open Positions
      </Typography>
      
      {positions.length === 0 ? (
        <Typography variant="body2" color="text.secondary">
          No open positions
        </Typography>
      ) : (
        <Box>
          {positions.map((pos, idx) => (
            <Box
              key={idx}
              sx={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                py: 1,
                borderBottom: idx < positions.length - 1 ? '1px solid rgba(255,255,255,0.1)' : 'none',
              }}
            >
              <Box>
                <Typography variant="body1" fontWeight="bold">
                  {pos.instrument}
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  {pos.units > 0 ? 'LONG' : 'SHORT'} {Math.abs(pos.units).toLocaleString()} units
                </Typography>
              </Box>
              <Box textAlign="right">
                <Typography
                  variant="body1"
                  fontWeight="bold"
                  color={pos.unrealizedPL >= 0 ? 'success.main' : 'error.main'}
                >
                  {pos.unrealizedPL >= 0 ? '+' : ''}{pos.unrealizedPL.toFixed(2)}
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  @ {pos.currentPrice.toFixed(5)}
                </Typography>
              </Box>
              <Button size="small" variant="outlined" color="error">
                Close
              </Button>
            </Box>
          ))}
        </Box>
      )}
    </Paper>
  );
};

// Main Dashboard Component
const ProfessionalDashboard = () => {
  const [activeTab, setActiveTab] = useState(0);
  const [systemStatus, setSystemStatus] = useState<SystemStatus>({
    training: 'idle',
    liveTrading: 'stopped',
    api: 'disconnected',
    gpu: { available: true, usage: 15, memory: 4.2, temperature: 45 },
  });
  const [account, setAccount] = useState<AccountInfo>({
    balance: 598687.94,
    equity: 601234.56,
    unrealizedPL: 2546.62,
    realizedPL: 12450.00,
    marginUsed: 1234.56,
    marginAvailable: 599999.38,
    currency: 'AUD',
  });
  const [positions, setPositions] = useState<Position[]>([
    { instrument: 'EUR_USD', units: 10000, unrealizedPL: 127.50, averagePrice: 1.0845, currentPrice: 1.0858 },
    { instrument: 'GBP_USD', units: -5000, unrealizedPL: -45.30, averagePrice: 1.2678, currentPrice: 1.2687 },
  ]);
  const [trainingMetrics, setTrainingMetrics] = useState<TrainingMetrics>({
    step: 58340,
    totalSteps: 100000,
    episode: 234,
    reward: 0.0523,
    loss: 0.0012,
    winRate: 0.675,
    eta: '12m 34s',
  });
  const [notification, setNotification] = useState<{ open: boolean; message: string; severity: 'success' | 'error' | 'info' }>({
    open: false,
    message: '',
    severity: 'info',
  });

  // WebSocket connection
  useEffect(() => {
    const ws = new WebSocket(WS_URL);

    ws.onopen = () => {
      setSystemStatus(prev => ({ ...prev, api: 'connected' }));
    };

    ws.onclose = () => {
      setSystemStatus(prev => ({ ...prev, api: 'disconnected' }));
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        // Handle different message types
        if (data.type === 'account') setAccount(data.payload);
        if (data.type === 'positions') setPositions(data.payload);
        if (data.type === 'training') setTrainingMetrics(data.payload);
        if (data.type === 'status') setSystemStatus(prev => ({ ...prev, ...data.payload }));
      } catch (e) {
        console.error('WebSocket message parse error:', e);
      }
    };

    return () => ws.close();
  }, []);

  // Handlers
  const handleStartTraining = () => {
    setSystemStatus(prev => ({ ...prev, training: 'running' }));
    setNotification({ open: true, message: 'Training started', severity: 'success' });
  };

  const handleStopTraining = () => {
    setSystemStatus(prev => ({ ...prev, training: 'idle' }));
    setNotification({ open: true, message: 'Training stopped', severity: 'info' });
  };

  const handlePauseTraining = () => {
    setSystemStatus(prev => ({ ...prev, training: 'paused' }));
  };

  return (
    <Box sx={{ flexGrow: 1, bgcolor: 'background.default', minHeight: '100vh', p: 2 }}>
      {/* Header */}
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
        <Typography variant="h4" fontWeight="bold">
          🚀 OANDA AI Trading Terminal
        </Typography>
        <Box>
          <IconButton>
            <Badge badgeContent={3} color="error">
              <Notifications />
            </Badge>
          </IconButton>
          <IconButton>
            <Settings />
          </IconButton>
        </Box>
      </Box>

      {/* Tab Navigation */}
      <Tabs value={activeTab} onChange={(_, v) => setActiveTab(v)} sx={{ mb: 2 }}>
        <Tab label="Dashboard" icon={<ShowChart />} iconPosition="start" />
        <Tab label="Training" icon={<Speed />} iconPosition="start" />
        <Tab label="Live Trading" icon={<AccountBalance />} iconPosition="start" />
        <Tab label="Models" icon={<Storage />} iconPosition="start" />
      </Tabs>

      {/* Dashboard Tab */}
      {activeTab === 0 && (
        <Grid container spacing={2}>
          {/* Metrics Row */}
          <Grid item xs={12} sm={6} md={3}>
            <MetricCard
              title="Balance"
              value={`$${account.balance.toLocaleString()}`}
              subtitle={account.currency}
              icon={<AccountBalance />}
              color="primary"
            />
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <MetricCard
              title="Equity"
              value={`$${account.equity.toLocaleString()}`}
              icon={<TrendingUp />}
              color="primary"
            />
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <MetricCard
              title="Unrealized P&L"
              value={`$${account.unrealizedPL >= 0 ? '+' : ''}${account.unrealizedPL.toLocaleString()}`}
              icon={account.unrealizedPL >= 0 ? <TrendingUp /> : <TrendingDown />}
              trend={account.unrealizedPL >= 0 ? 'up' : 'down'}
              color={account.unrealizedPL >= 0 ? 'success' : 'error'}
            />
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <MetricCard
              title="Today's P&L"
              value={`$${account.realizedPL >= 0 ? '+' : ''}${account.realizedPL.toLocaleString()}`}
              trend={account.realizedPL >= 0 ? 'up' : 'down'}
              color={account.realizedPL >= 0 ? 'success' : 'error'}
            />
          </Grid>

          {/* Main Content */}
          <Grid item xs={12} md={8}>
            <Paper sx={{ p: 2, height: 400, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <Typography color="text.secondary">
                [TradingView Chart Placeholder - Connect real data]
              </Typography>
            </Paper>
          </Grid>

          {/* Sidebar */}
          <Grid item xs={12} md={4}>
            <SystemStatusPanel status={systemStatus} />
          </Grid>

          {/* Positions */}
          <Grid item xs={12}>
            <PositionsPanel positions={positions} />
          </Grid>
        </Grid>
      )}

      {/* Training Tab */}
      {activeTab === 1 && (
        <Grid container spacing={2}>
          <Grid item xs={12} md={8}>
            <TrainingPanel
              metrics={trainingMetrics}
              onStart={handleStartTraining}
              onStop={handleStopTraining}
              onPause={handlePauseTraining}
              isRunning={systemStatus.training === 'running'}
            />
          </Grid>
          <Grid item xs={12} md={4}>
            <SystemStatusPanel status={systemStatus} />
          </Grid>
        </Grid>
      )}

      {/* Notifications */}
      <Snackbar
        open={notification.open}
        autoHideDuration={3000}
        onClose={() => setNotification(prev => ({ ...prev, open: false }))}
      >
        <Alert severity={notification.severity} onClose={() => setNotification(prev => ({ ...prev, open: false }))}>
          {notification.message}
        </Alert>
      </Snackbar>
    </Box>
  );
};

export default ProfessionalDashboard;
