import apiClient from './client';
import type { Strategy } from '@/types';

export async function getStrategies(params?: {
  search?: string;
  sort_by?: string;
  order?: string;
}): Promise<Strategy[]> {
  const response = await apiClient.get('/strategies', { params });
  return response.data;
}

export async function getStrategy(id: string): Promise<Strategy> {
  const response = await apiClient.get(`/strategies/${id}`);
  return response.data;
}

export async function createStrategy(data: {
  name: string;
  description?: string;
  method?: string;
  lookback_months?: number;
}): Promise<Strategy> {
  const response = await apiClient.post('/strategies', data);
  return response.data;
}
