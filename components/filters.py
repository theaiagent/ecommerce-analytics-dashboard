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
