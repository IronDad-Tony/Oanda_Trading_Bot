import React from 'react';
import { useSelector } from 'react-redux';
import { RootState } from '../store/store';
import { Box, Chip, Typography } from '@mui/material';
import { ConnectionStatus } from '../store/slices/systemSlice';

const GlobalStatus = () => {
  const status = useSelector((state: RootState) => state.system.connectionStatus);

  const getStatusChip = (status: ConnectionStatus) => {
    let color: "success" | "warning" | "error" | "default" = "default";
    let label = status.toUpperCase();

    switch (status) {
      case 'connected':
        color = 'success';
        break;
      case 'connecting':
        color = 'warning';
        break;
      case 'disconnected':
        color = 'error';
        break;
      case 'error':
        color = 'error';
        break;
    }

    return <Chip label={label} color={color} size="small" />;
  };

  return (
    <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end', height: '100%' }}>
      <Typography variant="body1" sx={{ mr: 1.5 }}>
        SYSTEM STATUS:
      </Typography>
      {getStatusChip(status)}
    </Box>
  );
};

export default GlobalStatus;
