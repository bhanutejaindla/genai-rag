import requests

class WeatherTool:
    def __init__(self, api_key):
        self.api_key = api_key

    def get_weather(self, city):
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={self.api_key}&units=metric"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            desc = data["weather"][0]["description"]
            temp = data["main"]["temp"]
            return f"Weather in {city}: {desc}, temperature: {temp}°C"
        return f"Could not fetch weather for {city}"
