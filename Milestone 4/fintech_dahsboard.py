import pandas as pd
import plotly.express as px
import dash
from dash import dcc, html
from dash.dependencies import Input, Output

def create_dashboard(data_path):
    # Load dataset
    df = pd.read_csv(data_path)

    # Initialize Dash app
    app = dash.Dash(__name__)
    app.title = "Fintech Loan Analysis Dashboard"
    loan_grade_percentages = df['grade_category_encoded'].value_counts(normalize=True).reset_index()
    loan_grade_percentages.columns = ['grade_category_encoded', 'percentage']
    loan_grade_percentages['percentage'] *= 100  # Convert to percentage
    # Layout
    app.layout = html.Div([
        html.Div([
            html.H1("Fintech Loan Analysis Dashboard", style={
                'textAlign': 'center',
                'color': '#FFFFFF',
                'fontFamily': 'Verdana, sans-serif',
                'fontSize': '40px',
                'backgroundColor': '#34495e',
                'padding': '20px',
                'borderRadius': '15px',
                'boxShadow': '0px 4px 10px rgba(0, 0, 0, 0.3)'
            }),
            html.H3("Created by: Shahenda Elsayed , ID: 52-23665", style={
                'textAlign': 'center',
                'color': '#3498db',
                'fontFamily': 'Verdana, sans-serif',
                'fontSize': '20px',
                'marginTop': '10px'
            }),
        ], style={'marginBottom': '30px'}),

        html.Br(),

        # 1. Loan Distribution by Grade
        html.Div([
            html.H2("1. Distribution of Loan Amounts Across Different Grades", style={'color': '#e74c3c', 'fontSize': '28px'}),
            dcc.Graph(
                figure=px.box(df, x='grade_category_encoded', y='loan_amount',
                               title="Loan Amount Distribution by Grade",
                               labels={'grade_category_encoded': 'Loan Grade', 'loan_amount': 'Loan Amount'},
                               color_discrete_sequence=["#8e44ad"])
            ),
        ], style={'backgroundColor': '#f9ebea', 'padding': '20px', 'borderRadius': '15px', 'marginBottom': '30px', 'boxShadow': '0px 4px 10px rgba(0, 0, 0, 0.2)'}),

        html.Br(),

        # 2. Loan Amount vs Annual Income Across States
        html.Div([
            html.H2("2. Loan Amount vs Annual Income Across States", style={'color': '#3498db', 'fontSize': '28px'}),
            dcc.Dropdown(
                id='state-dropdown',
                options=[{'label': state, 'value': state} for state in df['addr_state'].unique()] + [{'label': 'All', 'value': 'all'}],
                value='all',
                style={'width': '50%', 'marginBottom': '20px', 'color': '#34495e'}
            ),
            dcc.Graph(id='loan-vs-income-scatter'),
        ], style={'backgroundColor': '#eaf2f8', 'padding': '20px', 'borderRadius': '15px', 'marginBottom': '30px', 'boxShadow': '0px 4px 10px rgba(0, 0, 0, 0.2)'}),

        html.Br(),

        # 3. Loan Issuance Trend Over Months
        html.Div([
            html.H2("3. Trend of Loan Issuance Over Months", style={'color': '#2ecc71', 'fontSize': '28px'}),
            dcc.Dropdown(
                id='year-dropdown',
                options=[{'label': year, 'value': year} for year in pd.to_datetime(df['issue_date']).dt.year.unique()],
                value=pd.to_datetime(df['issue_date']).dt.year.min(),
                style={'width': '50%', 'marginBottom': '20px', 'color': '#34495e'}
            ),
            dcc.Graph(id='loan-trend-line'),
        ], style={'backgroundColor': '#e9f7ef', 'padding': '20px', 'borderRadius': '15px', 'marginBottom': '30px', 'boxShadow': '0px 4px 10px rgba(0, 0, 0, 0.2)'}),

        html.Br(),

          # 4. Top 5 States with Highest Average Loan Amount
        html.Div([
            html.H2("4. Top 5 States with the Highest Average Loan Amount", style={'color': '#f39c12', 'fontSize': '28px'}),
            dcc.Graph(
                figure=px.bar(df.groupby('addr_state')['loan_amount'].mean().reset_index()
                              .sort_values('loan_amount', ascending=False).head(5),
                              x='addr_state', y='loan_amount',
                              title="Top 5 States by Average Loan Amount",
                              labels={'addr_state': 'State', 'loan_amount': 'Average Loan Amount'},
                              color_discrete_sequence=["#d35400"])
            ),
        ], style={'backgroundColor': '#fef9e7', 'padding': '20px', 'borderRadius': '15px', 'marginBottom': '30px', 'boxShadow': '0px 4px 10px rgba(0, 0, 0, 0.2)'}),

        html.Br(),

        # 5. Percentage Distribution of Loan Grades
        html.Div([
            html.H2("5. Percentage Distribution of Loan Grades", style={'color': '#9b59b6', 'fontSize': '28px'}),
            dcc.Graph(
                figure=px.bar(loan_grade_percentages, x='grade_category_encoded', y='percentage',
                              title="Percentage Distribution of Loan Grades",
                              labels={'grade_category_encoded': 'Loan Grade', 'percentage': 'Percentage (%)'},
                              color_discrete_sequence=["#9b59b6"])
            ),
        ], style={'backgroundColor': '#f5eef8', 'padding': '20px', 'borderRadius': '15px', 'marginBottom': '30px', 'boxShadow': '0px 4px 10px rgba(0, 0, 0, 0.2)'}),

    ], style={'fontFamily': 'Verdana, sans-serif', 'backgroundColor': '#f4f6f7', 'padding': '30px'})

    # Callbacks
    @app.callback(
        Output('loan-vs-income-scatter', 'figure'),
        Input('state-dropdown', 'value')
    )
    def update_scatter(selected_state):
        if selected_state == 'all':
            filtered_df = df
        else:
            filtered_df = df[df['addr_state'] == selected_state]
        return px.scatter(
            filtered_df, x='annual_inc', y='loan_amount', color='loan_status',
            title="Loan Amount vs Annual Income",
            labels={'annual_inc': 'Annual Income', 'loan_amount': 'Loan Amount'},
            color_discrete_sequence=["#16a085", "#e74c3c"]
        )

    @app.callback(
        Output('loan-trend-line', 'figure'),
        Input('year-dropdown', 'value')
    )
    def update_trend(selected_year):
        df['issue_date'] = pd.to_datetime(df['issue_date'])
        filtered_df = df[df['issue_date'].dt.year == selected_year]
        trend = filtered_df.groupby(filtered_df['issue_date'].dt.month).size().reset_index(name='count')
        return px.line(
            trend, x='issue_date', y='count',
            title=f"Loan Issuance Trend for {selected_year}",
            labels={'issue_date': 'Month', 'count': 'Number of Loans'},
            line_shape='spline',
            color_discrete_sequence=["#27ae60"]
        )

    app.run_server(host='0.0.0.0', port=8050 , debug=False , threaded=True)