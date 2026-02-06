import axios from 'axios';

const apiClient = axios.create({
  baseURL: import.meta.env.VITE_API_URL || '/api',
  timeout: 120000, // 2 minutes - backtest can take time
  headers: {
    'Content-Type': 'application/json',
  },
});

export default apiClient;
