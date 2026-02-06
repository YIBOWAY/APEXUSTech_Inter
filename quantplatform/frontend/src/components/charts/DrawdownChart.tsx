import React, { useMemo } from 'react';
import ReactECharts from 'echarts-for-react';
import { useTheme } from '@/contexts/ThemeContext';
import type { EquityPoint } from '@/types';

interface DrawdownChartProps {
  data: EquityPoint[];
  height?: number | string;
  className?: string;
}

export const DrawdownChart: React.FC<DrawdownChartProps> = ({ 
  data, 
  height = 250,
  className 
}) => {
  const { theme } = useTheme();
  
  // Calculate Drawdown
  const drawdowns = useMemo(() => {
    let max = -Infinity;
    return data.map(d => {
      if (d.value > max) max = d.value;
      const dd = ((d.value - max) / max) * 100;
      return { date: d.date, value: parseFloat(dd.toFixed(2)) };
    });
  }, [data]);

  const option = useMemo(() => {
    const isDark = theme === 'dark';
    const textColor = isDark ? '#e2e8f0' : '#475569';
    const gridColor = isDark ? 'rgba(255, 255, 255, 0.05)' : 'rgba(0, 0, 0, 0.05)';

    return {
      backgroundColor: 'transparent',
      tooltip: {
        trigger: 'axis',
        valueFormatter: (value: number) => value.toFixed(2) + '%'
      },
      grid: {
        left: '2%',
        right: '2%',
        bottom: '5%',
        top: '5%',
        containLabel: true
      },
      xAxis: {
        type: 'category',
        boundaryGap: false,
        data: drawdowns.map(d => d.date),
        axisLine: { show: false },
        axisLabel: { color: textColor },
        splitLine: { show: false }
      },
      yAxis: {
        type: 'value',
        axisLine: { show: false },
        axisLabel: { color: textColor, formatter: '{value}%' },
        splitLine: { show: true, lineStyle: { color: gridColor } }
      },
      series: [
        {
          name: 'Drawdown',
          type: 'line',
          smooth: true,
          symbol: 'none',
          lineStyle: { width: 1, color: '#f43f5e' }, // Rose-500
          areaStyle: {
            color: {
              type: 'linear',
              x: 0, y: 0, x2: 0, y2: 1,
              colorStops: [{
                offset: 0, color: 'rgba(244, 63, 94, 0.5)'
              }, {
                offset: 1, color: 'rgba(244, 63, 94, 0.1)'
              }]
            }
          },
          data: drawdowns.map(d => d.value)
        }
      ]
    };
  }, [drawdowns, theme]);

  return (
    <div className={className} style={{ height }}>
      {/* @ts-ignore */}
      <ReactECharts 
        option={option} 
        style={{ height: '100%', width: '100%' }}
      />
    </div>
  );
};
