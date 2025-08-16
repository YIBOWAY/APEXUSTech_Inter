import nasdaqdatalink


nasdaqdatalink.ApiConfig.api_key = "SLKCCLcBMmxQnthU6Tb6"  # 替换为你的Key


data = nasdaqdatalink.get("WIKI/AAPL", rows=5, api_key = 'SLKCCLcBMmxQnthU6Tb6')
print(data)