import plotly.graph_objects as go
import pandas as pd
import numpy as np
from components.charts import (
    create_revenue_chart,
    create_category_chart,
    create_segments_chart,
    create_region_chart,
)


def _sample_orders():
    dates = pd.date_range("2025-03-01", periods=365, freq="D")
    categories = ["Electronics", "Clothing", "Home", "Sports", "Beauty"]
    regions = ["North America", "Europe", "Asia", "South America", "Middle East", "Africa"]
    n = 500
    return pd.DataFrame({
        "date": np.random.choice(dates, n),
        "total": np.random.uniform(10, 500, n),
        "category": np.random.choice(categories, n),
        "region": np.random.choice(regions, n),
        "customer_id": [f"C{i}" for i in np.random.randint(1, 200, n)],
        "status": np.random.choice(["Completed", "Pending", "Cancelled"], n),
    })


def _sample_customers():
    return pd.DataFrame({
        "customer_id": [f"C{i}" for i in range(200)],
        "segment": np.random.choice(["New", "Returning", "VIP"], 200),
    })


def test_revenue_chart():
    fig = create_revenue_chart(_sample_orders())
    assert isinstance(fig, go.Figure)


def test_category_chart():
    fig = create_category_chart(_sample_orders())
    assert isinstance(fig, go.Figure)


def test_segments_chart():
    fig = create_segments_chart(_sample_orders(), _sample_customers())
    assert isinstance(fig, go.Figure)


def test_region_chart():
    fig = create_region_chart(_sample_orders())
    assert isinstance(fig, go.Figure)
