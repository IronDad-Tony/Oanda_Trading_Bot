import React from 'react';
import { Button, Stack, Paper, Typography } from '@mui/material';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import PauseIcon from '@mui/icons-material/Pause';
import StopCircleIcon from '@mui/icons-material/StopCircle';
import apiService from '../services/apiService';
import { useDispatch } from 'react-redux';
import { addLog } from '../store/slices/logSlice';

const GlobalControls = () => {
  const dispatch = useDispatch();

  const handlePause = async () => {
    try {
      const res = await apiService.pauseTrading();
      dispatch(addLog({ level: 'SYSTEM', message: `Request to PAUSE trading sent. Response: ${res.message}` }));
    } catch (error: any) {
      dispatch(addLog({ level: 'ERROR', message: `Failed to pause trading: ${error.message}` }));
    }
  };

  const handleResume = async () => {
    try {
      const res = await apiService.resumeTrading();
      dispatch(addLog({ level: 'SYSTEM', message: `Request to RESUME trading sent. Response: ${res.message}` }));
    } catch (error: any) {
      dispatch(addLog({ level: 'ERROR', message: `Failed to resume trading: ${error.message}` }));
    }
  };

  const handleEmergencyStop = async () => {
    const confirmed = window.confirm(
      'ARE YOU SURE?\n\nThis will stop the trading bot and attempt to liquidate all open positions immediately.'
    );
    if (confirmed) {
      try {
        const res = await apiService.emergencyStop();
        dispatch(addLog({ level: 'SYSTEM', message: `EMERGENCY STOP initiated. Response: ${res.message}` }));
      } catch (error: any) {
        dispatch(addLog({ level: 'ERROR', message: `Failed to initiate emergency stop: ${error.message}` }));
      }
    } else {
      dispatch(addLog({ level: 'SYSTEM', message: 'Emergency stop cancelled.' }));
    }
  };


  return (
    <Paper sx={{ p: 2, height: '100%', display: 'flex', alignItems: 'center' }}>
        <Stack direction="row" spacing={1} alignItems="center">
            <Typography variant="body1">CONTROLS:</Typography>
            <Button variant="contained" color="warning" startIcon={<PauseIcon />} onClick={handlePause}>
                Pause
            </Button>
            <Button variant="contained" color="success" startIcon={<PlayArrowIcon />} onClick={handleResume}>
                Resume
            </Button>
            <Button variant="contained" color="error" startIcon={<StopCircleIcon />} onClick={handleEmergencyStop}>
                Emergency Stop
            </Button>
        </Stack>
    </Paper>
  );
};

export default GlobalControls;
