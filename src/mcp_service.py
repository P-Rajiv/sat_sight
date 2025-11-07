# src/mcp_service.py
from flask import Flask, request, jsonify
import requests
import wikipedia

app = Flask(__name__)

# --- Utility helpers ---
def fetch_weather(lat, lon):
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&daily=temperature_2m_max,temperature_2m_min,precipitation_sum&forecast_days=3&timezone=auto"
        r = requests.get(url, timeout=10)
        data = r.json().get("daily", {})
        temps = list(zip(data.get("temperature_2m_min", []), data.get("temperature_2m_max", [])))
        return {"forecast": temps[:3], "precipitation": data.get("precipitation_sum", [])[:3]}
    except Exception as e:
        return {"error": str(e)}

def fetch_wiki_summary(region):
    try:
        return {"summary": wikipedia.summary(region, sentences=2)}
    except Exception:
        return {"summary": f"No summary found for {region}"}

# --- Endpoints ---
@app.route("/earthdata", methods=["GET"])
def earthdata():
    lat = float(request.args.get("lat"))
    lon = float(request.args.get("lon"))
    data = fetch_weather(lat, lon)
    return jsonify(data)

@app.route("/wiki", methods=["GET"])
def wiki():
    region = request.args.get("region", "")
    data = fetch_wiki_summary(region)
    return jsonify(data)

if __name__ == "__main__":
    app.run(port=5001, debug=False)
