from dash import html
from components.kpi_cards import create_kpi_card, create_kpi_row
import pandas as pd
import numpy as np


def _sample_orders():
    dates = pd.date_range("2025-03-01", periods=365, freq="D")
    return pd.DataFrame({
        "date": np.random.choice(dates, 100),
        "total": np.random.uniform(10, 500, 100),
        "order_id": range(100),
        "customer_id": [f"C{i}" for i in np.random.randint(1, 50, 100)],
    })


def test_create_kpi_card():
    card = create_kpi_card("Revenue", "$12,345", "+5.2%", positive=True)
    assert card is not None


def test_create_kpi_row():
    orders = _sample_orders()
    row = create_kpi_row(orders)
    assert row is not None
