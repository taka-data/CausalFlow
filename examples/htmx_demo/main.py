from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
import causalflow as cf
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing

app = FastAPI()

# Global model state for demo purposes
data = fetch_california_housing()
X = data.data[:1000]
T = data.data[:1000, 0] # Use MedInc as treatment
Y = data.target[:1000]
model = cf.create_model(features=X, treatment=T, outcome=Y, feature_names=list(data.feature_names))

@app.get("/", response_class=HTMLResponse)
async def index():
    with open("examples/htmx_demo/index.html", "r") as f:
        return f.read()

@app.get("/plot/graph")
async def plot_graph():
    # Returns just the HTML fragment!
    return HTMLResponse(content=model.to_html(plot_type='graph'))

@app.get("/plot/importance")
async def plot_importance():
    results = model.estimate_effects(X[:100])
    # Returns just the HTML fragment!
    return HTMLResponse(content=results.to_html())

@app.get("/plot/dist")
async def plot_dist():
    return HTMLResponse(content=model.to_html(plot_type='effect_dist'))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
