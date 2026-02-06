import React, { useMemo } from 'react';
import ReactECharts from 'echarts-for-react';
import { useTheme } from '@/contexts/ThemeContext';
import type { EquityPoint } from '@/types';
import { format } from 'date-fns';

interface EquityChartProps {
  data: EquityPoint[];
  height?: number | string;
  className?: string;
  showBenchmark?: boolean;
}

export const EquityChart: React.FC<EquityChartProps> = ({ 
  data, 
  height = 400,
  className,
  showBenchmark = true
}) => {
  const { theme } = useTheme();
  
  const option = useMemo(() => {
    const isDark = theme === 'dark';
    const textColor = isDark ? '#e2e8f0' : '#475569';
    const gridColor = isDark ? 'rgba(255, 255, 255, 0.05)' : 'rgba(0, 0, 0, 0.05)';
    const tooltipBg = isDark ? 'rgba(15, 23, 42, 0.95)' : 'rgba(255, 255, 255, 0.95)';
    const tooltipBorder = isDark ? '#334155' : '#e2e8f0';

    return {
      backgroundColor: 'transparent',
      tooltip: {
        trigger: 'axis',
        backgroundColor: tooltipBg,
        borderColor: tooltipBorder,
        textStyle: { color: textColor },
        formatter: function (params: any[]) {
          const date = params[0].axisValue;
          let html = `<div class="font-mono text-sm font-semibold mb-1">${date}</div>`;
          params.forEach((param) => {
            const val = typeof param.value === 'number' ? param.value.toFixed(2) : param.value;
            html += `
              <div class="flex items-center justify-between gap-4 text-xs">
                <span style="color: ${param.color}">● ${param.seriesName}</span>
                <span class="font-mono font-medium">${val}</span>
              </div>
            `;
          });
          return html;
        }
      },
      legend: {
        bottom: 0,
        textStyle: { color: textColor },
        data: ['Strategy', showBenchmark ? 'Benchmark' : ''].filter(Boolean)
      },
      grid: {
        left: '2%',
        right: '2%',
        bottom: '10%',
        top: '5%',
        containLabel: true
      },
      xAxis: {
        type: 'category',
        boundaryGap: false,
        data: data.map(d => d.date),
        axisLine: { show: false },
        axisTick: { show: false },
        axisLabel: { color: textColor, margin: 12 },
        splitLine: { show: true, lineStyle: { color: gridColor } }
      },
      yAxis: {
        type: 'value',
        scale: true,
        axisLine: { show: false },
        axisTick: { show: false },
        axisLabel: { color: textColor, formatter: (val: number) => val.toFixed(0) },
        splitLine: { show: true, lineStyle: { color: gridColor } }
      },
      dataZoom: [
        {
          type: 'inside',
          start: 0,
          end: 100
        },
        {
          start: 0,
          end: 100,
          type: 'slider',
          height: 20,
          bottom: 30,
          borderColor: 'transparent',
          backgroundColor: isDark ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.05)',
          fillerColor: isDark ? 'rgba(129, 140, 248, 0.2)' : 'rgba(79, 70, 229, 0.2)',
          handleStyle: {
            color: isDark ? '#818cf8' : '#4f46e5',
            borderColor: 'transparent'
          },
          textStyle: { color: textColor }
        }
      ],
      series: [
        {
          name: 'Strategy',
          type: 'line',
          smooth: true,
          symbol: 'none',
          sampling: 'lttb',
          lineStyle: { width: 2, color: isDark ? '#818cf8' : '#4f46e5' }, // Indigo-400 : Indigo-600
          areaStyle: {
            color: {
              type: 'linear',
              x: 0, y: 0, x2: 0, y2: 1,
              colorStops: [{
                offset: 0, color: isDark ? 'rgba(129, 140, 248, 0.3)' : 'rgba(79, 70, 229, 0.3)'
              }, {
                offset: 1, color: isDark ? 'rgba(129, 140, 248, 0.01)' : 'rgba(79, 70, 229, 0.01)'
              }]
            }
          },
          data: data.map(d => d.value)
        },
        showBenchmark && {
          name: 'Benchmark',
          type: 'line',
          smooth: true,
          symbol: 'none',
          lineStyle: { width: 1.5, type: 'dashed', color: isDark ? '#64748b' : '#94a3b8' }, // Slate-500 : Slate-400
          data: data.map(d => d.benchmark)
        }
      ].filter(Boolean)
    };
  }, [data, theme, showBenchmark]);

  return (
    <div className={className} style={{ height }}>
      {/* @ts-ignore */}
      <ReactECharts 
        option={option} 
        style={{ height: '100%', width: '100%' }}
        opts={{ renderer: 'canvas' }} 
      />
    </div>
  );
};
