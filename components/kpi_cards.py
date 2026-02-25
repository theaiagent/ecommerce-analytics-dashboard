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
