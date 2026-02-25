from dash import dash_table
from components.charts import create_orders_table
import pandas as pd
import numpy as np


def test_create_orders_table():
    orders = pd.DataFrame({
        "order_id": ["ORD001", "ORD002"],
        "date": ["2025-06-01", "2025-06-02"],
        "customer_id": ["C001", "C002"],
        "category": ["Electronics", "Clothing"],
        "quantity": [1, 3],
        "total": [199.99, 59.97],
        "region": ["Europe", "Asia"],
        "status": ["Completed", "Pending"],
    })
    table = create_orders_table(orders)
    assert table is not None
