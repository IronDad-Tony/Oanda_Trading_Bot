import React, { useEffect, useRef } from 'react';
import { createChart, IChartApi, CandlestickData } from 'lightweight-charts';
import { Paper, Typography } from '@mui/material';

// Mock data for initial display
const mockCandlestickData: CandlestickData[] = [
  { time: '2023-11-15', open: 170.34, high: 172.5, low: 169.88, close: 172.22 },
  { time: '2023-11-16', open: 172.22, high: 174.8, low: 171.9, close: 174.5 },
  { time: '2023-11-17', open: 174.5, high: 175.1, low: 172.1, close: 173.9 },
  { time: '2023-11-18', open: 173.9, high: 176.5, low: 173.5, close: 176.0 },
  { time: '2023-11-19', open: 176.0, high: 178.2, low: 175.8, close: 177.9 },
  { time: '2023-11-20', open: 177.9, high: 180.0, low: 177.5, close: 179.5 },
];

const InteractiveChart = () => {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);

  useEffect(() => {
    if (!chartContainerRef.current) return;

    chartRef.current = createChart(chartContainerRef.current, {
      width: chartContainerRef.current.clientWidth,
      height: 360, // Set a fixed height
      layout: {
        background: { color: '#1e1e1e' },
        textColor: '#d1d4dc',
      },
      grid: {
        vertLines: { color: '#2b2b43' },
        horzLines: { color: '#2b2b43' },
      },
      timeScale: {
        borderColor: '#484848',
      },
    });

    const candlestickSeries = chartRef.current.addCandlestickSeries({
      upColor: '#26a69a',
      downColor: '#ef5350',
      borderDownColor: '#ef5350',
      borderUpColor: '#26a69a',
      wickDownColor: '#ef5350',
      wickUpColor: '#26a69a',
    });

    candlestickSeries.setData(mockCandlestickData);
    
    // Handle resize
    const handleResize = () => {
        if (chartRef.current && chartContainerRef.current) {
            chartRef.current.resize(chartContainerRef.current.clientWidth, 360);
        }
    };
    window.addEventListener('resize', handleResize);

    // Cleanup on unmount
    return () => {
      window.removeEventListener('resize', handleResize);
      if (chartRef.current) {
        chartRef.current.remove();
      }
    };
  }, []);

  return (
    <Paper sx={{ p: 2, height: '400px' }}>
        <Typography variant="h6" sx={{ mb: 1 }}>
            EUR_USD Chart (Mock Data)
        </Typography>
        <div ref={chartContainerRef} style={{ width: '100%', height: '360px' }} />
    </Paper>
  );
};

export default InteractiveChart;
