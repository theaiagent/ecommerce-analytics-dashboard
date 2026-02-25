import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
import pandas as pd

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


# Task 8: Callbacks
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


if __name__ == "__main__":
    app.run(debug=True)
