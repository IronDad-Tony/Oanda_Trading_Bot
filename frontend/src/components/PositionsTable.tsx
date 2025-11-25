import React from 'react';
import { useSelector, useDispatch } from 'react-redux';
import { RootState } from '../store/store';
import { Position } from '../store/slices/positionsSlice';
import {
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Typography,
  Button,
  Box,
} from '@mui/material';
import apiService from '../services/apiService';
import { addLog } from '../store/slices/logSlice';

const PnLCell = ({ value }: { value: number }) => {
  const color = value >= 0 ? 'success.main' : 'error.main';
  return (
    <TableCell align="right" sx={{ color, fontWeight: 'bold' }}>
      {value.toFixed(2)}
    </TableCell>
  );
};

const DirectionCell = ({ units }: { units: number }) => {
    const isLong = units > 0;
    const color = isLong ? 'success.main' : 'error.main';
    const label = isLong ? 'LONG' : 'SHORT';
    return (
        <TableCell sx={{ color, fontWeight: 'bold' }}>
            {label}
        </TableCell>
    );
}

const PositionsTable = () => {
  const { positions, lastUpdated } = useSelector((state: RootState) => state.positions);
  const dispatch = useDispatch();

  const handleClose = async (instrument: string) => {
    if (!window.confirm(`Are you sure you want to close the position for ${instrument}?`)) {
      return;
    }
    try {
      const res = await apiService.closePosition(instrument);
      dispatch(addLog({ level: 'SYSTEM', message: `Request to CLOSE position for ${instrument} sent. Response: ${res.message}` }));
    } catch (error: any) {
      dispatch(addLog({ level: 'ERROR', message: `Failed to close position for ${instrument}: ${error.message}` }));
    }
  };

  const getPositionDetails = (position: Position) => {
    const longUnits = parseInt(position.long.units);
    const shortUnits = parseInt(position.short.units);

    if (longUnits > 0) {
        return {
            direction: 'LONG',
            units: longUnits,
            avgPrice: parseFloat(position.long.averagePrice),
        }
    }
    if (shortUnits > 0) {
        return {
            direction: 'SHORT',
            units: shortUnits,
            avgPrice: parseFloat(position.short.averagePrice),
        }
    }
    return null;
  }

  return (
    <Paper sx={{ p: 2 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
        <Typography variant="h6">Open Positions</Typography>
        {lastUpdated && (
            <Typography variant="caption" color="text.secondary">
              Last Updated: {new Date(lastUpdated).toLocaleTimeString()}
            </Typography>
          )}
      </Box>
      <TableContainer>
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell>Instrument</TableCell>
              <TableCell>Direction</TableCell>
              <TableCell align="right">Size</TableCell>
              <TableCell align="right">Avg. Price</TableCell>
              <TableCell align="right">Unrealized P/L</TableCell>
              <TableCell align="center">Actions</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {positions.length === 0 ? (
              <TableRow>
                <TableCell colSpan={6} align="center">
                  <Typography color="text.secondary" sx={{ p: 3 }}>No Open Positions</Typography>
                </TableCell>
              </TableRow>
            ) : (
              positions.map((pos) => {
                const details = getPositionDetails(pos);
                if (!details) return null;

                return (
                  <TableRow key={pos.instrument}>
                    <TableCell component="th" scope="row">
                      {pos.instrument}
                    </TableCell>
                    <DirectionCell units={details.direction === 'LONG' ? details.units : -details.units} />
                    <TableCell align="right">{details.units}</TableCell>
                    <TableCell align="right">{details.avgPrice.toFixed(5)}</TableCell>
                    <PnLCell value={parseFloat(pos.unrealizedPL)} />
                    <TableCell align="center">
                      <Button variant="outlined" size="small" color="secondary" onClick={() => handleClose(pos.instrument)}>
                        Close
                      </Button>
                    </TableCell>
                  </TableRow>
                )
              })
            )}
          </TableBody>
        </Table>
      </TableContainer>
    </Paper>
  );
};

export default PositionsTable;
