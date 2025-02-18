import streamlit as st
import numpy as np
import plotly.graph_objects as go

# Set page config and theme
st.set_page_config(page_title="Bias-Variance Tradeoff Demo",layout="wide")

# Header section with controls
st.title("Bias-Variance Tradeoff Demo")

# Create two columns for the sliders
col1, col2 = st.columns(2)

with col1:
    polynomial_degree = st.slider("Model Complexity (Polynomial Degree)", 
                                min_value=1, 
                                max_value=15, 
                                value=1)

with col2:
    noise_level = st.slider("Noise Level", 
                          min_value=0.0, 
                          max_value=5.0, 
                          value=1.05, 
                          step=0.01)

# Generate data
def generate_true_function(x):
    return 0.1 * (x - 2) * (x - 6) * (x - 10) + 10

x_train = np.linspace(0, 12, 20)
noise = np.random.normal(0, noise_level, len(x_train))
y_true_train = generate_true_function(x_train)
y_train = y_true_train + noise

x_test = np.linspace(0, 12, 200)
y_true = generate_true_function(x_test)

# Fit model
coeffs = np.polyfit(x_train, y_train, polynomial_degree)
y_pred = np.polyval(coeffs, x_test)

# Display current model equation
st.markdown("### Current Model Equation")
equation = "y = "
for i, coef in enumerate(reversed(coeffs)):
    if i == 0:
        equation += f"{coef:.2f}"
    else:
        equation += f" + {coef:.2f}x^{i}"
st.code(equation)

# Calculate metrics
bias = np.mean(np.abs(y_true - y_pred))
variance = np.var(y_pred)

# Create main plot
fig_main = go.Figure()

# Add traces
fig_main.add_trace(go.Scatter(x=x_test, y=y_true, name="True Function", 
                             line=dict(color="blue", width=2)))
fig_main.add_trace(go.Scatter(x=x_train, y=y_train, name="Training Data",
                             mode="markers", marker=dict(color="red", size=8)))
fig_main.add_trace(go.Scatter(x=x_test, y=y_pred, name="Model Prediction",
                             line=dict(color="green", width=2)))

# Update layout for light theme
fig_main.update_layout(
    template="plotly",
    paper_bgcolor="white",
    plot_bgcolor="white",
    height=400,
    margin=dict(t=30),
    showlegend=True,
    legend=dict(
        yanchor="bottom",
        y=0.99,
        xanchor="right",
        x=0.99
    )
)

# Display main plot
st.plotly_chart(fig_main, use_container_width=True)

# Create separate row for metrics with columns
metric_col1, metric_col2 = st.columns(2)

# Bias gauge
with metric_col1:
    fig_bias = go.Figure(go.Indicator(
        mode="gauge+number",
        value=bias,
        title={'text': "Bias"},
        gauge={
            'axis': {'range': [0, 10]},
            'bar': {'color': "blue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
        }
    ))
    fig_bias.update_layout(
        paper_bgcolor="white",
        height=250,
        margin=dict(t=30, b=0)
    )
    st.plotly_chart(fig_bias, use_container_width=True)

# Variance gauge
with metric_col2:
    fig_variance = go.Figure(go.Indicator(
        mode="gauge+number",
        value=variance,
        title={'text': "Variance"},
        gauge={
            'axis': {'range': [0, 10]},
            'bar': {'color': "blue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
        }
    ))
    fig_variance.update_layout(
        paper_bgcolor="white",
        height=250,
        margin=dict(t=30, b=0)
    )
    st.plotly_chart(fig_variance, use_container_width=True)

# Add theoretical bias-variance tradeoff plot
def create_theoretical_plot():
    # Generate x values for model complexity
    x = np.arange(1, 16)  # Integer values from 1 to 15
    
    # Generate theoretical curves
    bias = 4 * np.exp(-0.3 * x)  # Decreasing exponential for bias
    variance = 0.3 * np.exp(0.3 * x - 2)  # Increasing exponential for variance
    total_error = bias + variance  # Total error
    irreducible_error = np.ones_like(x) * 0.5  # Constant irreducible error
    
    fig = go.Figure()
    
    # Add gray shaded regions for underfitting and overfitting
    fig.add_vrect(x0=1, x1=5, 
                  fillcolor="rgba(200,200,200,0.2)", 
                  layer="below", line_width=0,
                  annotation_text="underfitting zone",
                  annotation_position="top left")
    
    fig.add_vrect(x0=10, x1=15, 
                  fillcolor="rgba(200,200,200,0.2)", 
                  layer="below", line_width=0,
                  annotation_text="overfitting zone",
                  annotation_position="top right")
    
    # Add traces for each curve
    fig.add_trace(go.Scatter(x=x, y=total_error, name="Total Error",
                            line=dict(color='green', width=2)))
    
    fig.add_trace(go.Scatter(x=x, y=bias, name="Bias²",
                            line=dict(color='blue', width=2)))
    
    fig.add_trace(go.Scatter(x=x, y=variance, name="Variance",
                            line=dict(color='orange', width=2, dash='dash')))
    
    fig.add_trace(go.Scatter(x=x, y=irreducible_error, name="Irreducible Error",
                            line=dict(color='red', width=2, dash='dot')))
    
    # Update layout for light theme
    fig.update_layout(
        template="plotly",
        paper_bgcolor="white",
        plot_bgcolor="white",
        height=400,
        margin=dict(t=30, b=30, l=60, r=30),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        ),
        xaxis_title="Model Complexity",
        yaxis_title="Error",
        xaxis=dict(
            tickmode='linear',
            tick0=1,
            dtick=1,
            ticktext=list(range(1, 16)),
            tickvals=list(range(1, 16))
        ),
        title={
            'text': "Bias-Variance Tradeoff",
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        }
    )
    
    # Add vertical line for optimal model complexity
    fig.add_vline(x=7, line_dash="dash", line_color="gray", opacity=0.3,
                  annotation_text="optimal model complexity",
                  annotation_position="bottom")
    
    return fig

# Display theoretical plot
st.markdown("### Theoretical Bias-Variance Tradeoff")
theoretical_fig = create_theoretical_plot()
st.plotly_chart(theoretical_fig, use_container_width=True)