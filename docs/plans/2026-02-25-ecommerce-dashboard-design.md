# E-Commerce Analytics Dashboard — Design Document

**Date:** 2026-02-25
**Purpose:** Portfolio showcase project for Fiverr/Upwork profiles
**Approach:** Executive KPI Dashboard (dark-themed, visually striking)

---

## Overview

A single-page, dark-themed e-commerce analytics dashboard built with Python Dash. Designed to demonstrate both visual polish and analytical depth as a portfolio piece for freelancing platforms.

## Layout & Visual Structure

**Theme:** Dark background (#0f1117), card-based layout. Accent: electric blue (#636EFA) with Plotly's default palette. Bootstrap DARKLY theme via dash-bootstrap-components.

**Layout (top to bottom):**

1. **Header bar** — Title ("E-Commerce Analytics"), date range picker, category/region filter dropdowns
2. **KPI row** — 4 cards: Total Revenue, Total Orders, Active Customers, Avg Order Value. Each with sparkline + percent change indicator (green/red arrow)
3. **Charts row 1** — Revenue over time (area chart, 12 months) | Sales by category (horizontal bar)
4. **Charts row 2** — Customer segments (donut chart) | Orders by region (choropleth/treemap)
5. **Bottom** — Interactive data table of recent orders (sortable, searchable)

## Data Model & Generation

Fake data generated with Faker + NumPy/Pandas on app startup.

### Entities

- **Orders** (~5,000 records) — order_id, date (12 months), customer_id, product_id, quantity, unit_price, total, region, status
- **Products** (~50 items) — product_id, name, category (Electronics, Clothing, Home, Sports, Beauty), base_price
- **Customers** (~1,000 records) — customer_id, name, email, segment (New/Returning/VIP), join_date, region

### Built-in Patterns

- Seasonal sales spikes (holiday season, Black Friday)
- Electronics and Clothing as top categories
- VIP customers with higher order values
- Realistic distribution curves

### KPIs

- Revenue = sum of order totals
- Avg Order Value = revenue / order count
- Period-over-period % change for sparklines

## Tech Stack & Project Structure

### Dependencies

- dash + dash-bootstrap-components
- plotly
- pandas + numpy
- faker

### Structure

```
demo_2/
├── app.py              # Main Dash app, layout, callbacks
├── data/
│   └── generate.py     # Fake data generation functions
├── components/
│   ├── kpi_cards.py    # KPI card components
│   ├── charts.py       # All chart figure functions
│   └── filters.py      # Filter dropdowns & date picker
├── assets/
│   └── custom.css      # Dark theme overrides & polish
├── requirements.txt
└── README.md           # Setup instructions + screenshots
```

## Interactivity & Callbacks

### Filters

- Date range picker — filters all charts
- Category dropdown (multi-select)
- Region dropdown (multi-select)

### Behavior

- All filters update every component via Dash callbacks
- KPI cards recalculate on filtered data
- Sparklines show full 12-month trend with filtered period highlighted
- Hover tooltips on all charts
- Clicking a bar in category chart cross-filters other charts
- Data table supports sorting and text search

### Out of Scope

No login, export buttons, or settings panel. Showcase only.
