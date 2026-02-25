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
