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
