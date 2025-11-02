"""
Dash Dashboard for Stock Sentiment Analysis
"""

import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta

# Import existing modules
from sentiment.sentiment_cache import get_sentiment_cache


# Initialize Dash app with Dark theme (use provided Bootstrap dark theme)
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

# Custom CSS for additional styling
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            /* Minimal custom styling - let Bootstrap DARKLY theme handle most styling */
            .dashboard-header {
                background: linear-gradient(135deg, #375a7f 0%, #2c3e50 100%);
                padding: 2rem 0;
                margin-bottom: 2rem;
                border-bottom: 2px solid #375a7f;
            }
            .metric-card {
                border-radius: 8px;
                border-left: 4px solid #00bc8c;
                background-color: rgba(44, 62, 80, 0.6);
                padding: 1rem;
                transition: transform 0.2s;
            }
            .metric-card:hover {
                transform: translateY(-2px);
            }
            .watchlist-table {
                font-size: 0.85rem;
                background-color: rgba(44, 62, 80, 0.4);
                border-radius: 6px;
                padding: 0.5rem;
            }
            .watchlist-table table {
                margin-bottom: 0;
            }
            .watchlist-table td, .watchlist-table th {
                padding: 0.4rem 0.5rem;
                border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            }
            .chart-container {
                border-radius: 8px;
                padding: 1.5rem;
                margin-bottom: 1rem;
            }
            .sentiment-positive {
                color: #00bc8c;
                font-weight: bold;
            }
            .sentiment-negative {
                color: #e74c3c;
                font-weight: bold;
            }
            .sentiment-neutral {
                color: #f39c12;
                font-weight: bold;
            }
            /* Ensure charts have transparent background to blend with theme */
            .js-plotly-plot {
                background-color: transparent !important;
            }
            /* Date picker dark theme styling */
            .DateInput, .DateInput_input {
                background-color: #2c3e50 !important;
                color: #ecf0f1 !important;
                border-color: #34495e !important;
            }
            .DateInput_input__focused {
                background-color: #34495e !important;
                border-color: #3498db !important;
            }
            .SingleDatePicker, .SingleDatePickerInput {
                background-color: transparent !important;
            }
            .DayPicker {
                background-color: #2c3e50 !important;
                color: #ecf0f1 !important;
            }
            .DayPicker__horizontal {
                background-color: #2c3e50 !important;
            }
            .CalendarDay {
                background-color: #34495e !important;
                color: #ecf0f1 !important;
                border-color: #2c3e50 !important;
            }
            .CalendarDay__selected, .CalendarDay__selected_span {
                background-color: #3498db !important;
                border-color: #3498db !important;
            }
            .CalendarDay__hovered_span {
                background-color: #2980b9 !important;
            }
            /* Dropdown dark theme */
            .Select-control {
                background-color: #2c3e50 !important;
                border-color: #34495e !important;
                color: #ecf0f1 !important;
            }
            .Select-input, .Select-placeholder, .Select--single > .Select-control .Select-value {
                color: #ecf0f1 !important;
            }
            .Select-menu-outer {
                background-color: #2c3e50 !important;
                border-color: #34495e !important;
            }
            .Select-option {
                background-color: #2c3e50 !important;
                color: #ecf0f1 !important;
            }
            .Select-option.is-focused {
                background-color: #34495e !important;
            }
            .Select-option.is-selected {
                background-color: #3498db !important;
            }
            /* Multi-select tags */
            .Select-multi-value-wrapper {
                background-color: transparent !important;
            }
            .Select-value-label {
                color: #ecf0f1 !important;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# App layout with dark background
app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H1("Market Monitor Dashboard", className="text-center mb-0"),
                html.P("Professional Financial Analysis with AI-Powered Sentiment", className="text-center mb-0")
            ], className="dashboard-header")
        ])
    ]),
    
    # Market Overview - Indexes (Top row, packed)
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H5("Market Indexes", className="mb-2"),
                html.Div(id="market-indexes", children=[
                    html.P("Click 'Run Analysis' to load market data", className="text-muted")
                ])
            ], className="chart-container")
        ], width=12)
    ], className="mb-2"),
    
    # Main content - Sidebar + Charts + Watchlists
    dbc.Row([
        # Left Sidebar (Controls)
        dbc.Col([
            html.Div([
                html.H4("Controls", className="mb-3"),
                
                # Stock selection - Multi-select with defaults
                html.Label("Select Stocks:", className="fw-bold"),
                dcc.Dropdown(
                    id='ticker-dropdown',
                    options=[
                        {'label': 'Amazon.com Inc. (AMZN)', 'value': 'AMZN'},
                        {'label': 'Meta Platforms Inc. (META)', 'value': 'META'},
                        {'label': 'Apple Inc. (AAPL)', 'value': 'AAPL'},
                        {'label': 'Tesla Inc. (TSLA)', 'value': 'TSLA'},
                        {'label': 'Microsoft Corp. (MSFT)', 'value': 'MSFT'},
                        {'label': 'Alphabet Inc. (GOOGL)', 'value': 'GOOGL'},
                        {'label': 'NVIDIA Corp. (NVDA)', 'value': 'NVDA'},
                        {'label': 'Commonwealth Bank (CBA.AX)', 'value': 'CBA.AX'},
                        {'label': 'BHP Group Ltd. (BHP)', 'value': 'BHP'},
                    ],
                    value=['AMZN', 'META', 'CBA.AX'],  # Default 3 companies
                    multi=True,
                    className="mb-3"
                ),
                
                # Date range
                html.Label("Start Date:", className="fw-bold"),
                dcc.DatePickerSingle(
                    id='start-date',
                    date=datetime(2024, 1, 1),
                    max_date_allowed=datetime.now(),
                    className="mb-3"
                ),
                
                html.Label("End Date:", className="fw-bold"),
                dcc.DatePickerSingle(
                    id='end-date',
                    date=datetime.now(),
                    max_date_allowed=datetime.now(),
                    className="mb-3"
                ),
                
                # Analysis options
                html.Hr(),
                html.H5("Analysis Options", className="mb-3"),
                
                dbc.Checklist(
                    id='analysis-options',
                    options=[
                        {'label': ' Enable Sentiment Analysis', 'value': 'sentiment'},
                        {'label': ' Show Classification Results', 'value': 'classification'},
                    ],
                    value=['sentiment'],
                    className="mb-3"
                ),
                
                # Run button
                dbc.Button(
                    "Run Analysis",
                    id='run-button',
                    color="primary",
                    size="lg",
                    className="w-100"
                )
            ], className="sidebar")
        ], width=2),
        
        # Main Charts Area
        dbc.Col([
            dcc.Loading(
                id="loading",
                children=[html.Div([
                    html.H3("Welcome to the Market Monitor Dashboard"),
                    html.P("Select stocks and click 'Run Analysis' to begin."),
                    html.Div([
                        html.H4("Features:"),
                        html.Ul([
                            html.Li("📊 Real-time stock price analysis"),
                            html.Li("😊 AI-powered sentiment analysis"),
                            html.Li("📰 News sentiment tracking"),
                            html.Li("🎯 Classification predictions"),
                            html.Li("📈 Interactive charts and visualizations")
                        ])
                    ])
                ], id="analysis-results")],
                type="circle"
            )
        ], width=7),
        
        # Right Sidebar - Watchlists
        dbc.Col([
            html.Div([
                html.H5("Watchlists", className="mb-3"),
                html.Div(id="watchlists", children=[
                    html.Div([
                        html.H6("Liquid Mega Caps", className="mb-2"),
                        html.Div(id="watchlist-mega-caps", children=[
                            html.P("Loading...", className="text-muted small")
                        ], className="watchlist-table")
                    ], className="mb-3"),
                    html.Div([
                        html.H6("Theme ETFs", className="mb-2"),
                        html.Div(id="watchlist-etfs", children=[
                            html.P("Loading...", className="text-muted small")
                        ], className="watchlist-table")
                    ])
                ])
            ], className="chart-container")
        ], width=3)
    ])
], fluid=True)

# Helper function to normalize date strings (remove time component)
def _normalize_date_str(date_value):
    """Normalize date string to YYYY-MM-DD format, handling datetime strings."""
    if date_value is None:
        return None
    if isinstance(date_value, str):
        # Remove time component if present (e.g., "2025-11-01T10:47:09" -> "2025-11-01")
        return date_value.split('T')[0]
    elif hasattr(date_value, 'date'):
        return date_value.date().strftime("%Y-%m-%d")
    elif isinstance(date_value, datetime):
        return date_value.date().strftime("%Y-%m-%d")
    return date_value

# Helper function to fetch market indexes
def _fetch_market_indexes():
    """Fetch major market indexes with % change today."""
    indexes = ['SPY', 'QQQ', 'IWM', 'DIA']
    results = []
    try:
        from dataio.loading import load_stock_data_by_ticker
        today = datetime.now().date()
        start = (today - timedelta(days=7)).strftime("%Y-%m-%d")
        end = today.strftime("%Y-%m-%d")
        
        for idx in indexes:
            try:
                data = load_stock_data_by_ticker(idx, start_date=start, end_date=end, cache_dir='cache', max_years=1)
                if not data.empty and 'Close' in data.columns:
                    current = data['Close'].iloc[-1]
                    prev_close = data['Close'].iloc[0] if len(data) > 1 else current
                    pct_change = ((current - prev_close) / prev_close * 100) if prev_close > 0 else 0.0
                    volume = data['Volume'].iloc[-1] if 'Volume' in data.columns else 0
                    results.append({
                        'ticker': idx,
                        'price': current,
                        'pct_change': pct_change,
                        'volume': volume
                    })
            except Exception as e:
                print(f"Error loading {idx}: {e}")
                continue
    except Exception as e:
        print(f"Error in _fetch_market_indexes: {e}")
    return results

# Helper function to fetch watchlist stocks
def _fetch_watchlist_stocks(tickers, start_date, end_date):
    """Fetch watchlist data with % change today and 5d."""
    results = []
    try:
        from dataio.loading import load_stock_data_by_ticker
        # Normalize dates
        end_date = _normalize_date_str(end_date)
        today = datetime.now().date()
        start_recent = (today - timedelta(days=10)).strftime("%Y-%m-%d")
        
        for ticker in tickers:
            try:
                # Get recent data for today's change
                data_recent = load_stock_data_by_ticker(ticker, start_date=start_recent, end_date=end_date, cache_dir='cache', max_years=1)
                
                if not data_recent.empty and 'Close' in data_recent.columns:
                    current = data_recent['Close'].iloc[-1]
                    prev_close = data_recent['Close'].iloc[0] if len(data_recent) > 1 else current
                    pct_change_today = ((current - prev_close) / prev_close * 100) if prev_close > 0 else 0.0
                    
                    # 5-day change
                    if len(data_recent) >= 5:
                        close_5d_ago = data_recent['Close'].iloc[-6] if len(data_recent) > 6 else data_recent['Close'].iloc[0]
                        pct_change_5d = ((current - close_5d_ago) / close_5d_ago * 100) if close_5d_ago > 0 else 0.0
                    else:
                        pct_change_5d = pct_change_today
                    
                    results.append({
                        'ticker': ticker,
                        'pct_change_today': pct_change_today,
                        'pct_change_5d': pct_change_5d
                    })
            except Exception as e:
                print(f"Error loading {ticker} for watchlist: {e}")
                continue
    except Exception as e:
        print(f"Error in _fetch_watchlist_stocks: {e}")
    return results

# Market Indexes Callback
@app.callback(
    Output('market-indexes', 'children'),
    Input('run-button', 'n_clicks'),
    [Input('start-date', 'date'), Input('end-date', 'date')],
    prevent_initial_call=False
)
def update_market_indexes(n_clicks, start_date, end_date):
    """Update market indexes panel."""
    # Normalize dates to prevent parsing errors
    start_date = _normalize_date_str(start_date)
    end_date = _normalize_date_str(end_date)
    indexes = _fetch_market_indexes()
    if not indexes:
        return html.P("Click 'Run Analysis' to load market indexes", className="text-muted")
    
    cards = []
    for idx_data in indexes:
        color_class = "text-success" if idx_data['pct_change'] >= 0 else "text-danger"
        cards.append(
            dbc.Col([
                html.Div([
                    html.H6(idx_data['ticker'], className="mb-1"),
                    html.H5(f"{idx_data['pct_change']:.2f}%", className=f"mb-0 {color_class}"),
                    html.P(f"${idx_data['price']:.2f}", className="text-muted small mb-0")
                ], className="metric-card text-center p-2")
            ], width=3)
        )
    
    return dbc.Row(cards, className="g-2")

# Watchlists Callback
@app.callback(
    [Output('watchlist-mega-caps', 'children'),
     Output('watchlist-etfs', 'children')],
    Input('run-button', 'n_clicks'),
    [Input('start-date', 'date'), Input('end-date', 'date')],
    prevent_initial_call=False
)
def update_watchlists(n_clicks, start_date, end_date):
    """Update watchlists with % change metrics."""
    # Normalize dates to prevent parsing errors
    start_date = _normalize_date_str(start_date)
    end_date = _normalize_date_str(end_date)
    
    # Liquid Mega Caps
    mega_caps = ['NVDA', 'META', 'AMZN', 'MSFT', 'TSLA', 'AAPL', 'GOOGL']
    mega_data = _fetch_watchlist_stocks(mega_caps, start_date, end_date)
    
    # Theme ETFs
    etfs = ['JETS', 'GDX', 'SLX', 'SLV', 'TAN']
    etf_data = _fetch_watchlist_stocks(etfs, start_date, end_date)
    
    def _render_watchlist_table(data_list):
        """Render watchlist as compact table."""
        rows = []
        for item in data_list:
            today_color = "text-success" if item['pct_change_today'] >= 0 else "text-danger"
            day5_color = "text-success" if item['pct_change_5d'] >= 0 else "text-danger"
            rows.append(
                html.Tr([
                    html.Td(item['ticker'], className="fw-bold"),
                    html.Td(f"{item['pct_change_today']:.2f}%", className=today_color),
                    html.Td(f"{item['pct_change_5d']:.2f}%", className=day5_color)
                ])
            )
        if not rows:
            return html.P("No data available", className="text-muted small")
        return dbc.Table([
            html.Thead([
                html.Tr([
                    html.Th("Symbol", className="small"),
                    html.Th("% Today", className="small"),
                    html.Th("% 5d", className="small")
                ])
            ]),
            html.Tbody(rows)
        ], size="sm", bordered=False, hover=True, className="mb-0")
    
    mega_table = _render_watchlist_table(mega_data)
    etf_table = _render_watchlist_table(etf_data)
    
    return mega_table, etf_table

# Main Analysis Callback
@app.callback(
    Output('analysis-results', 'children'),
    Input('run-button', 'n_clicks'),
    [Input('ticker-dropdown', 'value'),
     Input('start-date', 'date'),
     Input('end-date', 'date'),
     Input('analysis-options', 'value')],
    prevent_initial_call=True  # Prevent auto-call that might cause callback errors
)
def run_analysis(n_clicks, tickers, start_date, end_date, options):
    try:
        # Handle initial call - return welcome message if dates not ready
        if start_date is None or end_date is None:
            return html.Div([
                html.H3("Welcome to the Stock Sentiment Analysis Dashboard"),
                html.P("Loading dashboard..."),
            ], className="text-center p-5")
        
        # Auto-load default stocks on first render (n_clicks is None)
        if n_clicks is None and (tickers is None or len(tickers) == 0):
            tickers = ['AMZN', 'META', 'CBA.AX']  # Default companies
        
        # Ensure tickers is a list
        if isinstance(tickers, str):
            tickers = [tickers]
        elif tickers is None or len(tickers) == 0:
            return html.Div([
                html.H3("Welcome to the Stock Sentiment Analysis Dashboard"),
                html.P("Select stocks and click 'Run Analysis' to begin."),
            ], className="text-center p-5")
        
        # Ensure options is a list
        if options is None:
            options = []
        
        # Map legacy/alias tickers to Yahoo Finance canonical symbols
        alias_map = {
            'FB': 'META',  # Facebook legacy ticker
        }
        # Normalize all tickers
        canonical_tickers = [alias_map.get(t, t) for t in tickers]
        # Convert dates to proper format (supports 'YYYY-MM-DD' and 'YYYY-MM-DDTHH:MM:SS')
        def _to_date_str_and_dt(value):
            try:
                if isinstance(value, str):
                    base = value.replace("Z", "")
                    # keep only date part for safety
                    date_part = base.split("T")[0]
                    dt = datetime.fromisoformat(date_part)
                    return date_part, dt.date()
                elif hasattr(value, 'date'):
                    dt = value.date()
                    return dt.strftime("%Y-%m-%d"), dt
                elif isinstance(value, datetime):
                    return value.strftime("%Y-%m-%d"), value.date()
                else:
                    # Fallback - use current date
                    today = datetime.now().date()
                    return today.strftime("%Y-%m-%d"), today
            except Exception as e:
                print(f"Error parsing date {value}: {e}")
                # Fallback to today
                today = datetime.now().date()
                return today.strftime("%Y-%m-%d"), today

        start_str, start_dt = _to_date_str_and_dt(start_date)
        end_str, end_dt = _to_date_str_and_dt(end_date)
        
        # Validate dates
        try:
            if start_dt > datetime.now().date():
                return dbc.Alert("Start date cannot be in the future.", color="warning")
            if end_dt > datetime.now().date():
                return dbc.Alert("End date cannot be in the future.", color="warning")
            if start_dt > end_dt:
                return dbc.Alert("Start date must be before end date.", color="warning")
        except Exception as e:
            print(f"Error validating dates: {e}")
            return dbc.Alert(f"Invalid date range: {str(e)}", color="warning")
        
        results = []
        
        # Create tabs for multiple stocks
        tab_list = []
        tab_content_list = []
        
        for ticker in canonical_tickers:
            try:
                print(f"[Dashboard] Loading data for {ticker} from {start_str} to {end_str}...")
                # Use load_stock_data_by_ticker() for efficient ticker-based caching
                from dataio.loading import load_stock_data_by_ticker
                stock_data = load_stock_data_by_ticker(
                    company=ticker,
                    start_date=start_str,
                    end_date=end_str,
                    cache_dir='cache'
                )
                
                # Validate data
                if stock_data is None:
                    print(f"[Dashboard] No data returned for {ticker}")
                    continue
                    
                if stock_data.empty: # type: ignore
                    print(f"[Dashboard] Empty dataframe for {ticker}")
                    continue
                
                if 'Close' not in stock_data.columns: # type: ignore
                    print(f"[Dashboard] Missing 'Close' column for {ticker}. Columns: {stock_data.columns.tolist()}")
                    continue
                    
                print(f"[Dashboard] Successfully loaded {len(stock_data)} rows for {ticker}")
                
                # Candlestick chart with volume (dark theme + transparency)
                from plotly.subplots import make_subplots
                
                # Create subplots: price chart on top, volume chart below
                fig_price = make_subplots(
                    rows=2, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.03,
                    row_heights=[0.7, 0.3],
                    subplot_titles=(f"{ticker} Stock Price", "Volume")
                )
                
                # Candlestick chart with transparency
                fig_price.add_trace(
                    go.Candlestick(
                        x=stock_data.index, # type: ignore
                        open=stock_data['Open'], # type: ignore
                        high=stock_data['High'], # type: ignore
                        low=stock_data['Low'], # type: ignore
                        close=stock_data['Close'], # type: ignore
                        name="Price",
                        increasing_line_color='rgba(38, 166, 154, 0.9)',  # Green with transparency
                        decreasing_line_color='rgba(239, 83, 80, 0.9)',  # Red with transparency
                        increasing_fillcolor='rgba(38, 166, 154, 0.6)',
                        decreasing_fillcolor='rgba(239, 83, 80, 0.6)',
                    ),
                    row=1, col=1
                )
                
                # Volume chart (colored bars with transparency)
                colors = [f'rgba(38, 166, 154, 0.7)' if stock_data['Close'].iloc[i] >= stock_data['Open'].iloc[i] else f'rgba(239, 83, 80, 0.7)' 
                          for i in range(len(stock_data))] # type: ignore
                fig_price.add_trace(
                    go.Bar(
                        x=stock_data.index, # type: ignore
                        y=stock_data['Volume'], # type: ignore
                        name="Volume",
                        marker_color=colors,
                        marker_line_width=0,
                        showlegend=False
                    ),
                    row=2, col=1
                )
                
                # Update layout with dark theme
                fig_price.update_layout(
                    title=f"{ticker} Stock Price Analysis",
                    height=600,
                    template="plotly_dark",  # Dark theme
                    plot_bgcolor='rgba(0, 0, 0, 0)',  # Transparent background
                    paper_bgcolor='rgba(0, 0, 0, 0)',  # Transparent paper
                    font=dict(color='#ffffff'),
                    xaxis_rangeslider_visible=False,
                    hovermode='x unified',
                )
                
                # Update axes with dark theme colors
                fig_price.update_xaxes(
                    title_text="Date", 
                    row=2, col=1,
                    gridcolor='rgba(255, 255, 255, 0.1)',
                    linecolor='rgba(255, 255, 255, 0.2)'
                )
                fig_price.update_yaxes(
                    title_text="Price ($)", 
                    row=1, col=1,
                    gridcolor='rgba(255, 255, 255, 0.1)',
                    linecolor='rgba(255, 255, 255, 0.2)'
                )
                fig_price.update_yaxes(
                    title_text="Volume", 
                    row=2, col=1,
                    gridcolor='rgba(255, 255, 255, 0.1)',
                    linecolor='rgba(255, 255, 255, 0.2)'
                )
                
                # Price metrics
                current_price = stock_data['Close'].iloc[-1] # type: ignore
                price_change = stock_data['Close'].iloc[-1] - stock_data['Close'].iloc[0] # type: ignore
                pct_change = (price_change / stock_data['Close'].iloc[0]) * 100 # type: ignore
                volatility = stock_data['Close'].pct_change().std() * np.sqrt(252) # type: ignore
                
                tab_content = html.Div([
                    html.Div([
                        html.H3(f"{ticker} Stock Price Analysis", className="mb-3"),
                        dcc.Graph(figure=fig_price, className="chart-container")
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                html.H4(f"${current_price:.2f}", className="text-primary"),
                                html.P("Current Price", className="mb-0")
                            ], className="metric-card text-center")
                        ], width=3),
                        dbc.Col([
                            html.Div([
                                html.H4(f"${price_change:.2f}", className="text-success" if price_change >= 0 else "text-danger"),
                                html.P("Total Change", className="mb-0")
                            ], className="metric-card text-center")
                        ], width=3),
                        dbc.Col([
                            html.Div([
                                html.H4(f"{pct_change:.2f}%", className="text-success" if pct_change >= 0 else "text-danger"),
                                html.P("Total Return", className="mb-0")
                            ], className="metric-card text-center")
                        ], width=3),
                        dbc.Col([
                            html.Div([
                                html.H4(f"{volatility:.2f}", className="text-info"),
                                html.P("Volatility", className="mb-0 text-white")
                            ], className="metric-card text-center")
                        ], width=3)
                    ], className="mt-3")
                ])
                
                tab_list.append(dbc.Tab(label=ticker, tab_id=f"tab-{ticker}"))
                tab_content_list.append(tab_content)
                
            except Exception as e:
                tab_content_list.append(
                    dbc.Alert(f"Error loading {ticker}: {str(e)}", color="danger")
                )
        
        # Add tabs if we have stocks - create tabs with tab panes
        if tab_list and len(tab_content_list) > 0:
            # Create tabs with their content
            tabs_with_content = []
            for i, ticker in enumerate(canonical_tickers):
                if i < len(tab_content_list):
                    tabs_with_content.append(
                        dbc.Tab(
                            tab_content_list[i],
                            label=ticker,
                            tab_id=f"tab-{ticker}"
                        )
                    )
            
            if tabs_with_content:
                results.append(
                    dbc.Tabs(
                        id="stock-tabs",
                        active_tab=f"tab-{canonical_tickers[0]}" if canonical_tickers else None,
                        children=tabs_with_content
                    )
                )
        elif len(canonical_tickers) > 0:
            # If we tried to load stocks but got no data
            results.append(
                dbc.Alert("No stock data could be loaded. Please check your date range and try again.", color="warning")
            )
        
        # 2. Sentiment Analysis (for first ticker only, or can extend to all)
        if 'sentiment' in options and canonical_tickers:
            try:
                # Show sentiment for first ticker (can extend to all)
                ticker_for_sentiment = canonical_tickers[0]
                sentiment_cache = get_sentiment_cache()
                news_df, daily_sentiment = sentiment_cache.build_daily_sentiment(ticker_for_sentiment, start_str, end_str)
                
                if not daily_sentiment.empty:
                    # Sentiment chart
                    fig_sentiment = go.Figure()
                    
                    # Add sentiment line
                    fig_sentiment.add_trace(go.Scatter(
                        x=daily_sentiment.index,
                        y=daily_sentiment['sentiment_mean'],
                        mode='lines+markers',
                        name='Sentiment Score',
                        line=dict(color='#ff7f0e', width=3),
                        marker=dict(size=8),
                        hovertemplate='<b>%{x}</b><br>Sentiment: %{y:.3f}<extra></extra>'
                    ))
                    
                    # Add zero line
                    fig_sentiment.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
                    
                    fig_sentiment.update_layout(
                        title=f"{ticker_for_sentiment} Daily Sentiment Analysis",
                        template="plotly_dark",  # Dark theme
                        plot_bgcolor='rgba(0, 0, 0, 0)',  # Transparent
                        paper_bgcolor='rgba(0, 0, 0, 0)',  # Transparent
                        font=dict(color='#ffffff'),
                        xaxis_title="Date",
                        yaxis_title="Sentiment Score",
                        height=400,
                        showlegend=False
                    )
                    
                    results.append(
                        html.Div([
                            html.H3("Sentiment Analysis", className="mb-3"),
                            dcc.Graph(figure=fig_sentiment, className="chart-container")
                        ])
                    )
                    
                    # Sentiment metrics
                    avg_sentiment = daily_sentiment['sentiment_mean'].mean()
                    total_news = daily_sentiment['news_count'].sum()
                    days_with_news = len(daily_sentiment)
                    avg_news_per_day = daily_sentiment['news_count'].mean()
                    
                    sentiment_metrics = dbc.Row([
                        dbc.Col([
                            html.Div([
                                html.H4(f"{avg_sentiment:.3f}", className="text-primary"),
                                html.P("Avg Sentiment", className="mb-0")
                            ], className="metric-card text-center")
                        ], width=3),
                        dbc.Col([
                            html.Div([
                                html.H4(f"{total_news}", className="text-info"),
                                html.P("Total News", className="mb-0")
                            ], className="metric-card text-center")
                        ], width=3),
                        dbc.Col([
                            html.Div([
                                html.H4(f"{days_with_news}", className="text-success"),
                                html.P("Days with News", className="mb-0")
                            ], className="metric-card text-center")
                        ], width=3),
                        dbc.Col([
                            html.Div([
                                html.H4(f"{avg_news_per_day:.1f}", className="text-warning"),
                                html.P("Avg News/Day", className="mb-0")
                            ], className="metric-card text-center")
                        ], width=3)
                    ])
                    
                    results.append(sentiment_metrics)
                    
                    # Recent news
                    if not news_df.empty:
                        recent_news = news_df.tail(5)[['date', 'title', 'sentiment_score']].copy()
                        
                        news_items = []
                        for _, row in recent_news.iterrows():
                            sentiment_class = "positive" if row['sentiment_score'] > 0.1 else "negative" if row['sentiment_score'] < -0.1 else "neutral"
                            emoji = "😊" if sentiment_class == "positive" else "😞" if sentiment_class == "negative" else "😐"
                            
                            news_items.append(
                                dbc.Card([
                                    dbc.CardBody([
                                        html.H6(f"{emoji} {row['date'].strftime('%Y-%m-%d')}", className="card-title"),
                                        html.P(row['title'], className="card-text")
                                    ])
                                ], className="mb-2")
                            )
                        
                        results.append(
                            html.Div([
                                html.H4("Recent News Headlines", className="mb-3"),
                                html.Div(news_items)
                            ])
                        )
                else:
                    results.append(
                        dbc.Alert("No sentiment data available for this period.", color="info")
                    )
                    
            except Exception as e:
                results.append(
                    dbc.Alert(f"Error in sentiment analysis: {str(e)}", color="danger")
                )
        
        # 3. Classification Results
        if 'classification' in options:
            results.append(
                html.Div([
                    html.H3("Classification Analysis", className="mb-3"),
                    dbc.Alert("Classification analysis would run here using your existing ClassificationEvaluator", color="info")
                ])
            )
        
        # Ensure we always return a valid component
        if len(results) == 0:
            return html.Div([
                html.H3("No results to display"),
                html.P("Please select stocks and click 'Run Analysis'."),
            ], className="text-center p-5")
        
        return html.Div(results)
        
    except Exception as e:
        import traceback
        error_msg = str(e)
        error_trace = traceback.format_exc()
        # Print to console for debugging
        print(f"\n{'='*60}")
        print("DASHBOARD CALLBACK ERROR:")
        print(f"{'='*60}")
        print(error_msg)
        print(f"\nTraceback:")
        print(error_trace)
        print(f"{'='*60}\n")
        
        # Return user-friendly error message
        return dbc.Alert([
            html.H5("Error occurred while processing your request"),
            html.P(error_msg, className="mb-2"),
            html.Hr(),
            html.P("Please check the server console for detailed error information.", className="text-muted small mb-0")
        ], color="danger")

if __name__ == '__main__':
    app.run(host="127.0.0.1", port=8050, debug=True, use_reloader=False)
