import React, { useMemo } from 'react';
import ReactECharts from 'echarts-for-react';
import { useTheme } from '@/contexts/ThemeContext';
import type { EquityPoint } from '@/types';
import { getMonth, getYear, lastDayOfMonth, format } from 'date-fns';

interface HeatmapChartProps {
  data: EquityPoint[];
  height?: number | string;
  className?: string;
}

export const HeatmapChart: React.FC<HeatmapChartProps> = ({ 
  data, 
  height = 300,
  className 
}) => {
  const { theme } = useTheme();

  // Calculate Monthly Returns
  const monthlyData = useMemo(() => {
    // Map: "2023-01" -> { start: 100, end: 105 }
    const monthMap = new Map<string, { start: number; end: number }>();
    
    // Sort by date just in case
    const sorted = [...data].sort((a, b) => a.date.localeCompare(b.date));

    sorted.forEach((pt, index) => {
      const d = new Date(pt.date);
      const key = format(d, 'yyyy-MM');
      
      if (!monthMap.has(key)) {
        // Find previous month's end value (or start of this month if first)
        const prevPt = index > 0 ? sorted[index - 1] : pt;
        monthMap.set(key, { start: prevPt.value, end: pt.value });
      } else {
        const entry = monthMap.get(key)!;
        entry.end = pt.value;
      }
    });

    // Convert to heatmap array: [x(MonthIndex), y(YearIndex), value]
    // But ECharts heatmap uses Category axis
    const years = Array.from(new Set(Array.from(monthMap.keys()).map(k => k.split('-')[0]))).sort();
    const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
    
    const chartData: any[] = [];
    
    monthMap.forEach((val, key) => {
      const [y, m] = key.split('-');
      const ret = ((val.end - val.start) / val.start) * 100;
      chartData.push([
        parseInt(m) - 1, // x: month index 0-11
        years.indexOf(y), // y: year index
        parseFloat(ret.toFixed(2)) // value
      ]);
    });

    return { years, months, chartData };
  }, [data]);

  const option = useMemo(() => {
    const isDark = theme === 'dark';
    const textColor = isDark ? '#e2e8f0' : '#475569';
    
    return {
      backgroundColor: 'transparent',
      tooltip: {
        position: 'top',
        formatter: (params: any) => {
          const { years, months } = monthlyData;
          const [mIndex, yIndex, val] = params.value;
          return `${years[yIndex]} ${months[mIndex]}: <b>${val}%</b>`;
        }
      },
      grid: {
        height: '80%',
        top: '10%'
      },
      xAxis: {
        type: 'category',
        data: monthlyData.months,
        splitArea: { show: true },
        axisLabel: { color: textColor }
      },
      yAxis: {
        type: 'category',
        data: monthlyData.years,
        splitArea: { show: true },
        axisLabel: { color: textColor }
      },
      visualMap: {
        min: -10,
        max: 10,
        calculable: false,
        orient: 'horizontal',
        left: 'center',
        bottom: 0,
        textStyle: { color: textColor },
        inRange: {
          color: isDark 
            ? ['#f43f5e', '#1e293b', '#10b981'] // Rose -> Slate -> Emerald
            : ['#f43f5e', '#f1f5f9', '#10b981'] // Rose -> Slate-100 -> Emerald
        }
      },
      series: [{
        name: 'Monthly Return',
        type: 'heatmap',
        data: monthlyData.chartData,
        label: {
          show: true,
          color: isDark ? '#fff' : '#000',
          formatter: (p: any) => p.value[2] + '%'
        },
        itemStyle: {
          borderColor: isDark ? '#0f172a' : '#fff',
          borderWidth: 2,
          borderRadius: 4
        }
      }]
    };
  }, [monthlyData, theme]);

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
