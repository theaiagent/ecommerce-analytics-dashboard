import pandas as pd
from data.generate import generate_products, generate_customers, generate_orders


def test_generate_products():
    df = generate_products()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 50
    assert set(df.columns) == {"product_id", "name", "category", "base_price"}
    assert set(df["category"].unique()).issubset(
        {"Electronics", "Clothing", "Home", "Sports", "Beauty"}
    )


def test_generate_customers():
    df = generate_customers()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1000
    assert "segment" in df.columns
    assert set(df["segment"].unique()).issubset({"New", "Returning", "VIP"})


def test_generate_orders():
    products = generate_products()
    customers = generate_customers()
    df = generate_orders(products, customers)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 5000
    expected_cols = {
        "order_id", "date", "customer_id", "product_id",
        "quantity", "unit_price", "total", "region", "status",
    }
    assert expected_cols.issubset(set(df.columns))
    assert df["total"].min() > 0


def test_seasonal_pattern():
    """Orders should spike in Nov-Dec (holiday season)."""
    products = generate_products()
    customers = generate_customers()
    orders = generate_orders(products, customers)
    orders["month"] = pd.to_datetime(orders["date"]).dt.month
    holiday = orders[orders["month"].isin([11, 12])]
    other = orders[~orders["month"].isin([11, 12])]
    avg_holiday = len(holiday) / 2
    avg_other = len(other) / 10
    assert avg_holiday > avg_other, "Holiday months should have more orders"
