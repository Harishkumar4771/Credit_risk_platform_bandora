import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

TEAL = "#0F5C5C"
BRICK = "#B4452F"
GREEN = "#2F6B4F"
INK = "#1F2933"
MUTED = "#6B7280"
LINE = "#E7E2DA"


def _layout(title=None, height=300):
    return dict(title=dict(text=title or ""), height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter", color=INK), margin=dict(l=12, r=12, t=42 if title else 12, b=18), showlegend=False)


def risk_meter(probability, threshold):
    figure = go.Figure(go.Bar(x=[probability], y=["Applicant"], orientation="h", marker_color=BRICK if probability >= threshold else GREEN, hovertemplate="Probability: %{x:.1%}<extra></extra>"))
    figure.update_layout(**_layout(height=125), xaxis=dict(range=[0, 1], tickformat=".0%", gridcolor=LINE, title=None), yaxis=dict(showticklabels=False), shapes=[dict(type="line", x0=threshold, x1=threshold, y0=-0.5, y1=0.5, line=dict(color=TEAL, width=3))], annotations=[dict(x=threshold, y=1.1, xref="x", yref="paper", text=f"65% cutoff", showarrow=False, font=dict(color=TEAL, size=12))])
    return figure


def model_bars(probabilities, threshold):
    names = list(probabilities)
    values = list(probabilities.values())
    colors = [BRICK if value >= threshold else GREEN for value in values]
    figure = go.Figure(go.Bar(x=values, y=names, orientation="h", marker_color=colors, text=[f"{value:.1%}" for value in values], textposition="outside", cliponaxis=False, hovertemplate="%{y}: %{x:.1%}<extra></extra>"))
    figure.update_layout(**_layout("What each model says", height=max(170, 55 * len(names))), xaxis=dict(range=[0, 1], tickformat=".0%", gridcolor=LINE), yaxis=dict(autorange="reversed"), shapes=[dict(type="line", x0=threshold, x1=threshold, y0=-0.5, y1=len(names) - 0.5, line=dict(color=TEAL, width=2, dash="dot"))])
    return figure


def signed_shap_bar(frame):
    frame = frame.sort_values("shap")
    frame = frame.copy()
    frame["axis_label"] = frame.apply(lambda row: f"{row['label']} ({row['display']})", axis=1)
    colors = [BRICK if value >= 0 else GREEN for value in frame["shap"]]
    figure = go.Figure(go.Bar(x=frame["shap"], y=frame["axis_label"], orientation="h", marker_color=colors, hovertemplate="%{y}<br>SHAP value: %{x:.3f}<extra></extra>"))
    layout = _layout(height=max(300, 46 * len(frame)))
    layout["margin"] = dict(l=220, r=36, t=12, b=42)
    figure.update_layout(**layout, xaxis=dict(title="Pushes risk down     Pushes risk up", zeroline=True, zerolinecolor=LINE, gridcolor=LINE), yaxis=dict(automargin=True, title=""))
    return figure


def line_what_if(frame, current):
    figure = px.line(frame, x="interest", y="probability", markers=True)
    figure.update_traces(line_color=TEAL, marker_color=TEAL)
    figure.add_scatter(x=[current["interest"]], y=[current["probability"]], mode="markers", marker=dict(size=11, color=BRICK), name="Current")
    figure.update_layout(**_layout("Probability as interest changes", height=270), xaxis_title="Interest rate (%)", yaxis_title="Probability", yaxis_tickformat=".0%", legend=dict(orientation="h", y=1.15))
    return figure


def global_importance(frame):
    frame = frame.sort_values("importance")
    figure = go.Figure(go.Bar(x=frame["importance"], y=frame["label"], orientation="h", marker_color=TEAL, text=[f"{value:.3f}" for value in frame["importance"]], textposition="outside", cliponaxis=False))
    figure.update_layout(**_layout("What matters most overall", height=max(330, 35 * len(frame))), xaxis_title="Mean absolute contribution", yaxis=dict(automargin=True))
    return figure


def decile_chart():
    deciles = ["Safest 10%", "2", "3", "4", "5", "6", "7", "8", "9", "Riskiest 10%"]
    rates = [14.0, 34.8, 50.8, 60.7, 67.4, 74.6, 78.7, 82.5, 85.6, 91.7]
    figure = go.Figure(go.Bar(x=deciles, y=rates, marker_color=TEAL, text=[f"{rate:.1f}%" for rate in rates], textposition="outside", cliponaxis=False))
    figure.update_layout(**_layout("Observed default rate by risk decile", height=340), yaxis=dict(range=[0, 105], ticksuffix="%", gridcolor=LINE), xaxis_title="Risk decile")
    return figure


def fairness_chart():
    countries = ["Estonia", "Finland", "Spain", "Slovakia"]
    rates = [16.2, 62.7, 74.5, 0]
    labels = ["16.2%", "62.7%", "74.5%", "Not audited"]
    colors = [BRICK, BRICK, BRICK, "#B7B0A6"]
    figure = go.Figure(go.Bar(x=rates, y=countries, orientation="h", marker_color=colors, text=labels, textposition="outside", cliponaxis=False, hovertemplate="%{y}: %{text}<extra></extra>"))
    figure.update_layout(**_layout("Good borrowers flagged by country", height=275), xaxis=dict(range=[0, 85], ticksuffix="%", gridcolor=LINE), yaxis=dict(autorange="reversed"))
    return figure
