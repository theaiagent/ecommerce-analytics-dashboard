import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from dash import dash_table

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
