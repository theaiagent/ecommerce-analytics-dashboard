# E-Commerce Analytics Dashboard — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a dark-themed, single-page e-commerce analytics dashboard in Python Dash as a freelancing portfolio piece.

**Architecture:** Modular Dash app with separate data generation, component, and chart modules. Fake data generated at startup via Faker/NumPy. All charts update reactively through Dash callbacks driven by filter controls.

**Tech Stack:** Python, Dash, dash-bootstrap-components (DARKLY theme), Plotly, Pandas, NumPy, Faker

---

## Task 1: Project Scaffolding & Dependencies

**Files:**
- Create: `requirements.txt`
- Create: `app.py`
- Create: `data/__init__.py`
- Create: `components/__init__.py`
- Create: `assets/custom.css`

**Step 1: Create requirements.txt**

```txt
dash==2.18.2
dash-bootstrap-components==1.6.0
plotly==5.24.1
pandas==2.2.3
numpy==2.1.3
faker==33.1.0
```

**Step 2: Create empty module directories**

```bash
mkdir -p data components assets
touch data/__init__.py components/__init__.py
```

**Step 3: Create minimal app.py that runs**

```python
import dash
from dash import html
import dash_bootstrap_components as dbc

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
app.title = "E-Commerce Analytics"

app.layout = dbc.Container(
    [html.H1("E-Commerce Analytics Dashboard", className="text-center my-4")],
    fluid=True,
)

if __name__ == "__main__":
    app.run(debug=True)
```

**Step 4: Create placeholder custom.css**

```css
body {
    background-color: #0f1117 !important;
}
```

**Step 5: Install dependencies and verify app runs**

```bash
pip install -r requirements.txt
python app.py
```

Expected: App starts on http://127.0.0.1:8050, shows title on dark background.

**Step 6: Commit**

```bash
git add requirements.txt app.py data/ components/ assets/
git commit -m "feat: scaffold project with Dash + DARKLY theme"
```

---

## Task 2: Fake Data Generation

**Files:**
- Create: `data/generate.py`
- Create: `tests/test_generate.py`

**Step 1: Create tests directory**

```bash
mkdir -p tests
touch tests/__init__.py
```

**Step 2: Write failing tests for data generation**

Create `tests/test_generate.py`:

```python
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
```

**Step 3: Run tests to verify they fail**

```bash
pytest tests/test_generate.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'data.generate'`

**Step 4: Implement data/generate.py**

```python
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
                start_date=start_date, end_date=end_date
            ),
            "region": np.random.choice(REGIONS),
        })
    return pd.DataFrame(customers)


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

    product_indices = np.random.choice(len(products), size=n, p=CATEGORY_WEIGHTS_FOR_ORDERS(products))
    chosen_products = products.iloc[product_indices]

    quantities = np.random.randint(1, 6, size=n)
    unit_prices = chosen_products["base_price"].values
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


def CATEGORY_WEIGHTS_FOR_ORDERS(products: pd.DataFrame) -> np.ndarray:
    """Weight each product by its category weight for order selection."""
    cat_weight_map = dict(zip(CATEGORIES, CATEGORY_WEIGHTS))
    weights = products["category"].map(cat_weight_map).values
    return weights / weights.sum()


def generate_all_data() -> dict:
    """Generate all datasets. Returns dict with 'products', 'customers', 'orders' DataFrames."""
    products = generate_products()
    customers = generate_customers()
    orders = generate_orders(products, customers)
    return {"products": products, "customers": customers, "orders": orders}
```

**Step 5: Run tests to verify they pass**

```bash
pytest tests/test_generate.py -v
```

Expected: All 4 tests PASS.

**Step 6: Commit**

```bash
git add data/generate.py tests/
git commit -m "feat: add fake e-commerce data generation with seasonal patterns"
```

---

## Task 3: KPI Card Components

**Files:**
- Create: `components/kpi_cards.py`
- Create: `tests/test_kpi_cards.py`

**Step 1: Write failing test**

Create `tests/test_kpi_cards.py`:

```python
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
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_kpi_cards.py -v
```

Expected: FAIL — `ModuleNotFoundError`

**Step 3: Implement components/kpi_cards.py**

```python
from dash import html, dcc
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import numpy as np


def create_sparkline(series: pd.Series) -> go.Figure:
    """Create a tiny sparkline figure for a KPI card."""
    fig = go.Figure(
        go.Scatter(
            y=series.values,
            mode="lines",
            line=dict(color="#636EFA", width=2),
            fill="tozeroy",
            fillcolor="rgba(99, 110, 250, 0.1)",
        )
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=40,
        width=120,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        showlegend=False,
    )
    return fig


def create_kpi_card(
    title: str, value: str, change: str, positive: bool, sparkline_fig: go.Figure = None
) -> dbc.Card:
    """Create a single KPI card with title, value, change indicator, and optional sparkline."""
    change_color = "#00cc96" if positive else "#ef553b"
    arrow = "\u25b2" if positive else "\u25bc"

    card_content = [
        html.P(title, className="kpi-title mb-1",
               style={"color": "#8a8d93", "fontSize": "0.85rem", "fontWeight": "500"}),
        html.H3(value, className="kpi-value mb-1",
                 style={"color": "#ffffff", "fontWeight": "700"}),
        html.Span(
            f"{arrow} {change}",
            style={"color": change_color, "fontSize": "0.85rem", "fontWeight": "600"},
        ),
    ]

    if sparkline_fig:
        card_content.append(
            dcc.Graph(figure=sparkline_fig, config={"displayModeBar": False},
                      style={"marginTop": "8px"})
        )

    return dbc.Card(
        dbc.CardBody(card_content, style={"padding": "1.25rem"}),
        className="kpi-card",
        style={
            "backgroundColor": "#1a1c23",
            "border": "1px solid #2a2d35",
            "borderRadius": "12px",
        },
    )


def create_kpi_row(orders: pd.DataFrame) -> dbc.Row:
    """Create the full KPI row from orders data."""
    orders["date"] = pd.to_datetime(orders["date"])
    total_revenue = orders["total"].sum()
    total_orders = len(orders)
    active_customers = orders["customer_id"].nunique()
    avg_order_value = total_revenue / total_orders if total_orders > 0 else 0

    # Monthly aggregation for sparklines and period-over-period change
    monthly = orders.set_index("date").resample("ME")
    rev_monthly = monthly["total"].sum()
    ord_monthly = monthly["order_id"].count()

    def pct_change(series):
        if len(series) < 2 or series.iloc[-2] == 0:
            return 0.0
        return ((series.iloc[-1] - series.iloc[-2]) / series.iloc[-2]) * 100

    rev_change = pct_change(rev_monthly)
    ord_change = pct_change(ord_monthly)

    cards = [
        create_kpi_card(
            "Total Revenue",
            f"${total_revenue:,.0f}",
            f"{abs(rev_change):.1f}%",
            positive=rev_change >= 0,
            sparkline_fig=create_sparkline(rev_monthly),
        ),
        create_kpi_card(
            "Total Orders",
            f"{total_orders:,}",
            f"{abs(ord_change):.1f}%",
            positive=ord_change >= 0,
            sparkline_fig=create_sparkline(ord_monthly),
        ),
        create_kpi_card(
            "Active Customers",
            f"{active_customers:,}",
            "",
            positive=True,
        ),
        create_kpi_card(
            "Avg Order Value",
            f"${avg_order_value:,.2f}",
            "",
            positive=True,
        ),
    ]

    return dbc.Row(
        [dbc.Col(card, xs=12, sm=6, lg=3) for card in cards],
        className="g-3 mb-4",
    )
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_kpi_cards.py -v
```

Expected: All 2 tests PASS.

**Step 5: Commit**

```bash
git add components/kpi_cards.py tests/test_kpi_cards.py
git commit -m "feat: add KPI card components with sparklines"
```

---

## Task 4: Chart Components

**Files:**
- Create: `components/charts.py`
- Create: `tests/test_charts.py`

**Step 1: Write failing tests**

Create `tests/test_charts.py`:

```python
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
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_charts.py -v
```

Expected: FAIL — `ModuleNotFoundError`

**Step 3: Implement components/charts.py**

```python
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

DARK_TEMPLATE = "plotly_dark"
PAPER_BG = "rgba(0,0,0,0)"
PLOT_BG = "rgba(26,28,35,1)"
GRID_COLOR = "rgba(255,255,255,0.05)"


def _base_layout() -> dict:
    return dict(
        template=DARK_TEMPLATE,
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(color="#c5c6c9"),
        margin=dict(l=40, r=20, t=40, b=40),
        xaxis=dict(gridcolor=GRID_COLOR),
        yaxis=dict(gridcolor=GRID_COLOR),
    )


def create_revenue_chart(orders: pd.DataFrame) -> go.Figure:
    """Monthly revenue area chart."""
    df = orders.copy()
    df["date"] = pd.to_datetime(df["date"])
    monthly = df.set_index("date").resample("ME")["total"].sum().reset_index()
    monthly.columns = ["Month", "Revenue"]

    fig = px.area(
        monthly, x="Month", y="Revenue",
        color_discrete_sequence=["#636EFA"],
    )
    fig.update_layout(**_base_layout(), title="Revenue Over Time")
    fig.update_traces(
        fill="tozeroy",
        fillcolor="rgba(99, 110, 250, 0.15)",
        line=dict(width=2),
    )
    return fig


def create_category_chart(orders: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart of sales by category."""
    cat_sales = orders.groupby("category")["total"].sum().sort_values()

    fig = px.bar(
        x=cat_sales.values, y=cat_sales.index,
        orientation="h",
        color_discrete_sequence=["#636EFA"],
    )
    fig.update_layout(**_base_layout(), title="Sales by Category")
    fig.update_traces(marker_cornerradius=6)
    return fig


def create_segments_chart(orders: pd.DataFrame, customers: pd.DataFrame) -> go.Figure:
    """Donut chart of revenue by customer segment."""
    merged = orders.merge(customers[["customer_id", "segment"]], on="customer_id", how="left")
    seg_revenue = merged.groupby("segment")["total"].sum().reset_index()

    fig = px.pie(
        seg_revenue, values="total", names="segment",
        hole=0.5,
        color_discrete_sequence=["#636EFA", "#EF553B", "#00CC96"],
    )
    fig.update_layout(**_base_layout(), title="Revenue by Customer Segment")
    return fig


def create_region_chart(orders: pd.DataFrame) -> go.Figure:
    """Treemap of orders by region."""
    region_data = orders.groupby("region")["total"].sum().reset_index()

    fig = px.treemap(
        region_data, path=["region"], values="total",
        color="total",
        color_continuous_scale="Blues",
    )
    fig.update_layout(**_base_layout(), title="Revenue by Region")
    fig.update_layout(coloraxis_showscale=False)
    return fig
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_charts.py -v
```

Expected: All 4 tests PASS.

**Step 5: Commit**

```bash
git add components/charts.py tests/test_charts.py
git commit -m "feat: add chart components (revenue, category, segments, region)"
```

---

## Task 5: Filter Components

**Files:**
- Create: `components/filters.py`
- Create: `tests/test_filters.py`

**Step 1: Write failing test**

Create `tests/test_filters.py`:

```python
from components.filters import create_filter_bar


def test_create_filter_bar():
    categories = ["Electronics", "Clothing", "Home"]
    regions = ["North America", "Europe"]
    bar = create_filter_bar(categories, regions)
    assert bar is not None
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_filters.py -v
```

Expected: FAIL

**Step 3: Implement components/filters.py**

```python
from dash import dcc, html
import dash_bootstrap_components as dbc
from datetime import date
from dateutil.relativedelta import relativedelta


def create_filter_bar(categories: list, regions: list) -> dbc.Row:
    """Create the filter bar with date picker, category, and region dropdowns."""
    today = date.today()
    year_ago = today - relativedelta(months=12)

    return dbc.Row(
        [
            dbc.Col(
                [
                    html.Label("Date Range", className="filter-label",
                               style={"color": "#8a8d93", "fontSize": "0.8rem", "marginBottom": "4px"}),
                    dcc.DatePickerRange(
                        id="date-filter",
                        start_date=year_ago,
                        end_date=today,
                        display_format="MMM D, YYYY",
                        style={"width": "100%"},
                    ),
                ],
                xs=12, md=4,
            ),
            dbc.Col(
                [
                    html.Label("Category", className="filter-label",
                               style={"color": "#8a8d93", "fontSize": "0.8rem", "marginBottom": "4px"}),
                    dcc.Dropdown(
                        id="category-filter",
                        options=[{"label": c, "value": c} for c in categories],
                        multi=True,
                        placeholder="All Categories",
                        style={"backgroundColor": "#1a1c23", "color": "#c5c6c9"},
                    ),
                ],
                xs=12, md=4,
            ),
            dbc.Col(
                [
                    html.Label("Region", className="filter-label",
                               style={"color": "#8a8d93", "fontSize": "0.8rem", "marginBottom": "4px"}),
                    dcc.Dropdown(
                        id="region-filter",
                        options=[{"label": r, "value": r} for r in regions],
                        multi=True,
                        placeholder="All Regions",
                        style={"backgroundColor": "#1a1c23", "color": "#c5c6c9"},
                    ),
                ],
                xs=12, md=4,
            ),
        ],
        className="g-3 mb-4 p-3",
        style={
            "backgroundColor": "#1a1c23",
            "borderRadius": "12px",
            "border": "1px solid #2a2d35",
        },
    )
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_filters.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add components/filters.py tests/test_filters.py
git commit -m "feat: add filter bar component (date, category, region)"
```

---

## Task 6: Data Table Component

**Files:**
- Modify: `components/charts.py` (add table function)
- Create: `tests/test_data_table.py`

**Step 1: Write failing test**

Create `tests/test_data_table.py`:

```python
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
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_data_table.py -v
```

Expected: FAIL

**Step 3: Add create_orders_table to components/charts.py**

Append to `components/charts.py`:

```python
from dash import dash_table


def create_orders_table(orders: pd.DataFrame) -> dash_table.DataTable:
    """Create an interactive data table for recent orders."""
    display_cols = ["order_id", "date", "customer_id", "category",
                    "quantity", "total", "region", "status"]
    df = orders[display_cols].tail(100).copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df["total"] = df["total"].apply(lambda x: f"${x:,.2f}")

    return dash_table.DataTable(
        data=df.to_dict("records"),
        columns=[{"name": col.replace("_", " ").title(), "id": col} for col in display_cols],
        sort_action="native",
        filter_action="native",
        page_size=15,
        style_header={
            "backgroundColor": "#1a1c23",
            "color": "#636EFA",
            "fontWeight": "600",
            "border": "1px solid #2a2d35",
        },
        style_data={
            "backgroundColor": "#0f1117",
            "color": "#c5c6c9",
            "border": "1px solid #2a2d35",
        },
        style_filter={
            "backgroundColor": "#1a1c23",
            "color": "#c5c6c9",
        },
        style_table={"borderRadius": "12px", "overflow": "hidden"},
    )
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_data_table.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add components/charts.py tests/test_data_table.py
git commit -m "feat: add interactive orders data table"
```

---

## Task 7: Assemble Layout in app.py

**Files:**
- Modify: `app.py`

**Step 1: Replace app.py with full layout**

```python
import dash
from dash import html, dcc
import dash_bootstrap_components as dbc

from data.generate import generate_all_data
from components.kpi_cards import create_kpi_row
from components.charts import (
    create_revenue_chart,
    create_category_chart,
    create_segments_chart,
    create_region_chart,
    create_orders_table,
)
from components.filters import create_filter_bar

# Generate data at startup
data = generate_all_data()
orders = data["orders"]
products = data["products"]
customers = data["customers"]

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
app.title = "E-Commerce Analytics"

app.layout = dbc.Container(
    [
        # Header
        html.H2(
            "E-Commerce Analytics",
            className="text-center my-4",
            style={"color": "#ffffff", "fontWeight": "700"},
        ),
        # Filters
        create_filter_bar(
            categories=sorted(orders["category"].unique()),
            regions=sorted(orders["region"].unique()),
        ),
        # KPI Cards
        html.Div(id="kpi-row", children=create_kpi_row(orders)),
        # Charts Row 1
        dbc.Row(
            [
                dbc.Col(
                    dcc.Graph(id="revenue-chart", figure=create_revenue_chart(orders)),
                    xs=12, lg=7,
                ),
                dbc.Col(
                    dcc.Graph(id="category-chart", figure=create_category_chart(orders)),
                    xs=12, lg=5,
                ),
            ],
            className="g-3 mb-4",
        ),
        # Charts Row 2
        dbc.Row(
            [
                dbc.Col(
                    dcc.Graph(id="segments-chart",
                              figure=create_segments_chart(orders, customers)),
                    xs=12, lg=5,
                ),
                dbc.Col(
                    dcc.Graph(id="region-chart", figure=create_region_chart(orders)),
                    xs=12, lg=7,
                ),
            ],
            className="g-3 mb-4",
        ),
        # Data Table
        html.H5("Recent Orders", style={"color": "#ffffff", "marginBottom": "12px"}),
        create_orders_table(orders),
    ],
    fluid=True,
    style={"maxWidth": "1400px", "padding": "0 24px"},
)

if __name__ == "__main__":
    app.run(debug=True)
```

**Step 2: Run the app and verify visually**

```bash
python app.py
```

Expected: Full dashboard visible at http://127.0.0.1:8050 with all sections.

**Step 3: Run all tests**

```bash
pytest tests/ -v
```

Expected: All tests PASS.

**Step 4: Commit**

```bash
git add app.py
git commit -m "feat: assemble full dashboard layout with all components"
```

---

## Task 8: Dash Callbacks (Interactivity)

**Files:**
- Modify: `app.py` — add callbacks

**Step 1: Add callbacks to app.py**

Add below the layout, before `if __name__`:

```python
from dash.dependencies import Input, Output


@app.callback(
    [
        Output("kpi-row", "children"),
        Output("revenue-chart", "figure"),
        Output("category-chart", "figure"),
        Output("segments-chart", "figure"),
        Output("region-chart", "figure"),
    ],
    [
        Input("date-filter", "start_date"),
        Input("date-filter", "end_date"),
        Input("category-filter", "value"),
        Input("region-filter", "value"),
    ],
)
def update_dashboard(start_date, end_date, selected_categories, selected_regions):
    filtered = orders.copy()
    filtered["date"] = pd.to_datetime(filtered["date"])

    if start_date and end_date:
        filtered = filtered[
            (filtered["date"] >= start_date) & (filtered["date"] <= end_date)
        ]
    if selected_categories:
        filtered = filtered[filtered["category"].isin(selected_categories)]
    if selected_regions:
        filtered = filtered[filtered["region"].isin(selected_regions)]

    return (
        create_kpi_row(filtered),
        create_revenue_chart(filtered),
        create_category_chart(filtered),
        create_segments_chart(filtered, customers),
        create_region_chart(filtered),
    )
```

Also add `import pandas as pd` at the top of app.py if not present.

**Step 2: Test interactivity manually**

```bash
python app.py
```

Expected: Changing filters updates all charts and KPIs.

**Step 3: Commit**

```bash
git add app.py
git commit -m "feat: add filter callbacks for interactive dashboard"
```

---

## Task 9: CSS Polish

**Files:**
- Modify: `assets/custom.css`

**Step 1: Write polished CSS**

```css
/* Base */
body {
    background-color: #0f1117 !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* KPI Cards hover effect */
.kpi-card {
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.kpi-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 20px rgba(99, 110, 250, 0.15);
}

/* Graph containers */
.js-plotly-plot .plotly .main-svg {
    border-radius: 12px;
}

/* Dropdown styling for dark theme */
.Select-control {
    background-color: #1a1c23 !important;
    border-color: #2a2d35 !important;
}
.Select-menu-outer {
    background-color: #1a1c23 !important;
    border-color: #2a2d35 !important;
}
.Select-value-label, .Select-placeholder {
    color: #c5c6c9 !important;
}
.VirtualizedSelectOption {
    background-color: #1a1c23;
    color: #c5c6c9;
}
.VirtualizedSelectFocusedOption {
    background-color: #2a2d35;
}

/* Date picker dark overrides */
.DateInput_input {
    background-color: #1a1c23 !important;
    color: #c5c6c9 !important;
    border-color: #2a2d35 !important;
    font-size: 0.85rem;
}
.DateRangePickerInput {
    background-color: #1a1c23 !important;
    border: 1px solid #2a2d35 !important;
    border-radius: 6px;
}
.DateRangePickerInput_arrow svg {
    fill: #636EFA;
}

/* DataTable scrollbar */
.dash-spreadsheet-container::-webkit-scrollbar {
    height: 6px;
}
.dash-spreadsheet-container::-webkit-scrollbar-track {
    background: #1a1c23;
}
.dash-spreadsheet-container::-webkit-scrollbar-thumb {
    background: #2a2d35;
    border-radius: 3px;
}

/* Responsive tweaks */
@media (max-width: 768px) {
    .kpi-value {
        font-size: 1.2rem !important;
    }
}
```

**Step 2: Verify visual polish**

```bash
python app.py
```

Expected: Hover effects on KPI cards, properly styled dropdowns and date pickers.

**Step 3: Commit**

```bash
git add assets/custom.css
git commit -m "feat: add dark theme CSS polish and hover effects"
```

---

## Task 10: README & Final Touches

**Files:**
- Create: `README.md`

**Step 1: Write README**

```markdown
# E-Commerce Analytics Dashboard

A dark-themed, interactive e-commerce analytics dashboard built with Python and Dash. Features real-time filtering, KPI tracking, and multiple visualization types.

![Dashboard Preview](docs/screenshot.png)

## Features

- **KPI Cards** — Revenue, Orders, Customers, Avg Order Value with sparklines
- **Interactive Charts** — Revenue trends, category breakdown, customer segments, regional analysis
- **Real-time Filtering** — Date range, category, and region filters update all visualizations
- **Data Table** — Sortable and searchable order history
- **Dark Theme** — Modern, professional design

## Tech Stack

- **Python** — Core language
- **Dash** — Web framework
- **Plotly** — Interactive charts
- **Pandas / NumPy** — Data processing
- **Faker** — Realistic sample data generation

## Quick Start

```bash
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:8050 in your browser.

## Project Structure

```
├── app.py              # Main app with layout and callbacks
├── data/
│   └── generate.py     # Fake e-commerce data generation
├── components/
│   ├── kpi_cards.py    # KPI card components
│   ├── charts.py       # Chart and table components
│   └── filters.py      # Filter controls
├── assets/
│   └── custom.css      # Dark theme styling
└── tests/              # Unit tests
```

## Sample Data

Generates ~5,000 orders across 50 products and 1,000 customers with realistic patterns:
- Seasonal holiday sales spikes
- Category-weighted product distribution
- VIP customer behavior modeling
```

**Step 2: Run full test suite one last time**

```bash
pytest tests/ -v
```

Expected: All tests PASS.

**Step 3: Commit**

```bash
git add README.md
git commit -m "docs: add README with project overview and setup instructions"
```

---

## Summary

| Task | Description | Files |
|------|-------------|-------|
| 1 | Project scaffolding | `requirements.txt`, `app.py`, dirs |
| 2 | Fake data generation | `data/generate.py`, tests |
| 3 | KPI card components | `components/kpi_cards.py`, tests |
| 4 | Chart components | `components/charts.py`, tests |
| 5 | Filter components | `components/filters.py`, tests |
| 6 | Data table | `components/charts.py` (append), tests |
| 7 | Assemble layout | `app.py` |
| 8 | Callbacks (interactivity) | `app.py` |
| 9 | CSS polish | `assets/custom.css` |
| 10 | README | `README.md` |
