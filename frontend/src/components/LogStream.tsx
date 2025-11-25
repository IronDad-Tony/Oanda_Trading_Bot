import React from 'react';
import { useSelector } from 'react-redux';
import { RootState } from '../store/store';
import { LogEntry } from '../store/slices/logSlice';
import { Box, Typography, List, ListItemText } from '@mui/material';

const getLogLevelColor = (level: LogEntry['level']) => {
  switch (level) {
    case 'ERROR':
      return 'error.main';
    case 'WARN':
      return 'warning.main';
    case 'SYSTEM':
      return 'primary.main';
    case 'INFO':
    case 'DEBUG':
    default:
      return 'text.secondary';
  }
};

const LogStream = () => {
  const logs = useSelector((state: RootState) => state.logs.entries);

  return (
    <>
      <Typography variant="h6" sx={{ p: 2, pb: 1 }}>
        Event Log
      </Typography>
      <Box sx={{ flexGrow: 1, overflowY: 'auto', px: 2 }}>
        <List dense>
          {logs.map((log, index) => (
            <ListItemText
              key={index}
              disableTypography
              primary={
                <Typography variant="body2" component="div">
                  <Box component="span" sx={{ color: 'text.disabled', mr: 1 }}>
                    {new Date(log.timestamp).toLocaleTimeString()}
                  </Box>
                  <Box component="span" sx={{ color: getLogLevelColor(log.level), fontWeight: 'bold', mr: 1 }}>
                    [{log.level}]
                  </Box>
                  <Box component="span" sx={{ color: 'text.primary' }}>
                    {log.message}
                  </Box>
                </Typography>
              }
            />
          ))}
        </List>
      </Box>
    </>
  );
};

export default LogStream;
