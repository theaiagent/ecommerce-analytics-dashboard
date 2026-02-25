import numpy as np
import pandas as pd
from faker import Faker

fake = Faker()
Faker.seed(42)
np.random.seed(42)

CATEGORIES = ["Electronics", "Clothing", "Home", "Sports", "Beauty"]
REGIONS = ["North America", "Europe", "Asia", "South America", "Middle East", "Africa"]
SEGMENTS = ["New", "Returning", "VIP"]
STATUSES = ["Completed", "Pending", "Cancelled"]

# Category weights: Electronics and Clothing are top sellers
CATEGORY_WEIGHTS = [0.30, 0.25, 0.20, 0.15, 0.10]

# Monthly order distribution weights (index 0=Jan, 11=Dec)
# Nov and Dec get ~2x weight for holiday spike
MONTHLY_WEIGHTS = np.array([0.07, 0.06, 0.07, 0.08, 0.08, 0.09,
                            0.08, 0.08, 0.09, 0.08, 0.11, 0.11])


def generate_products(n: int = 50) -> pd.DataFrame:
    products = []
    for i in range(n):
        cat = np.random.choice(CATEGORIES, p=CATEGORY_WEIGHTS)
        price_ranges = {
            "Electronics": (50, 1200),
            "Clothing": (15, 200),
            "Home": (20, 500),
            "Sports": (10, 300),
            "Beauty": (5, 150),
        }
        lo, hi = price_ranges[cat]
        products.append({
            "product_id": f"P{i+1:04d}",
            "name": fake.catch_phrase(),
            "category": cat,
            "base_price": round(np.random.uniform(lo, hi), 2),
        })
    return pd.DataFrame(products)


def generate_customers(n: int = 1000) -> pd.DataFrame:
    customers = []
    segment_weights = [0.50, 0.35, 0.15]  # New, Returning, VIP
    end_date = pd.Timestamp.now()
    start_date = end_date - pd.DateOffset(months=24)

    for i in range(n):
        seg = np.random.choice(SEGMENTS, p=segment_weights)
        customers.append({
            "customer_id": f"C{i+1:05d}",
            "name": fake.name(),
            "email": fake.email(),
            "segment": seg,
            "join_date": fake.date_between(
                start_date=start_date, end_date=end_date,
            ),
            "region": np.random.choice(REGIONS),
        })
    return pd.DataFrame(customers)


def _category_weights_for_orders(products: pd.DataFrame) -> np.ndarray:
    """Weight each product by its category weight for order selection."""
    cat_weight_map = dict(zip(CATEGORIES, CATEGORY_WEIGHTS))
    weights = products["category"].map(cat_weight_map).values
    return weights / weights.sum()


def generate_orders(
    products: pd.DataFrame,
    customers: pd.DataFrame,
    n: int = 5000,
) -> pd.DataFrame:
    end_date = pd.Timestamp.now()
    start_date = end_date - pd.DateOffset(months=12)

    # Generate dates with seasonal weighting
    months = np.random.choice(
        range(12), size=n, p=MONTHLY_WEIGHTS / MONTHLY_WEIGHTS.sum()
    )
    dates = []
    for m in months:
        month = start_date + pd.DateOffset(months=int(m))
        day = np.random.randint(1, 28)  # safe day range
        dates.append(month.replace(day=day))

    # VIP customers order more frequently
    vip_ids = customers[customers["segment"] == "VIP"]["customer_id"].values
    regular_ids = customers[customers["segment"] != "VIP"]["customer_id"].values

    customer_ids = []
    for _ in range(n):
        if np.random.random() < 0.30:  # 30% of orders from VIPs (15% of customers)
            customer_ids.append(np.random.choice(vip_ids))
        else:
            customer_ids.append(np.random.choice(regular_ids))

    product_indices = np.random.choice(
        len(products), size=n, p=_category_weights_for_orders(products)
    )
    chosen_products = products.iloc[product_indices]

    quantities = np.random.randint(1, 6, size=n)
    unit_prices = chosen_products["base_price"].values.copy()
    # Add some price variation (+/- 10%)
    unit_prices = unit_prices * np.random.uniform(0.9, 1.1, size=n)
    unit_prices = np.round(unit_prices, 2)
    totals = np.round(quantities * unit_prices, 2)

    orders = pd.DataFrame({
        "order_id": [f"ORD{i+1:06d}" for i in range(n)],
        "date": dates,
        "customer_id": customer_ids,
        "product_id": chosen_products["product_id"].values,
        "category": chosen_products["category"].values,
        "quantity": quantities,
        "unit_price": unit_prices,
        "total": totals,
        "region": np.random.choice(REGIONS, size=n),
        "status": np.random.choice(STATUSES, size=n, p=[0.80, 0.12, 0.08]),
    })
    return orders.sort_values("date").reset_index(drop=True)


def generate_all_data() -> dict:
    """Generate all datasets. Returns dict with 'products', 'customers', 'orders' DataFrames."""
    products = generate_products()
    customers = generate_customers()
    orders = generate_orders(products, customers)
    return {"products": products, "customers": customers, "orders": orders}
