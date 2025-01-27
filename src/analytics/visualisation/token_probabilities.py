import dash
from dash import dcc, html, Input, Output
import pandas as pd
import plotly.express as px

df = pd.read_csv("/Users/mannes/thesis/complex-task-gen/src/machine_learning/output/pipeline_output.csv")


# Initialize the Dash app
app = dash.Dash(__name__)

# Layout for the Dash app
app.layout = html.Div([
    html.H1("Token Log Probability Visualizer"),
    dcc.Dropdown(
        id="question-dropdown",
        options=[{"label": q, "value": q} for q in df["question"]],
        placeholder="Select a question",
    ),
    html.Div(id="token-output", style={"marginTop": 20}),
    dcc.Graph(id="log-p-graph"),
])


@app.callback(
    [Output("token-output", "children"),
     Output("log-p-graph", "figure")],
    [Input("question-dropdown", "value")]
)
def update_visualization(selected_question):
    if selected_question is None:
        return "Select a question to see details.", {}

    # Filter the dataframe for the selected question
    row = df[df["question"] == selected_question].iloc[0]

    # Extract and validate tokens and log probabilities
    tokens = row["tokenized_input"]
    log_probs = row["prompt_answer_log_p"]

    # Ensure tokens and log_probs are lists
    if isinstance(tokens, list) and len(tokens) == 1:
        tokens = tokens[0]  # Flatten nested list if needed
    if isinstance(log_probs, list) and len(log_probs) == 1:
        log_probs = log_probs[0]  # Flatten nested list if needed

    # Validate lengths of tokens and log_probs
    if len(tokens) != len(log_probs):
        return f"Mismatch in lengths: {len(tokens)} tokens and {len(log_probs)} log probabilities.", {}

    # Create a token-log probability mapping
    token_log_mapping = [
        f"Token: '{token}' | Log Probability: {log_p}"
        for token, log_p in zip(tokens, log_probs)
    ]
    token_details = html.Ul([html.Li(item) for item in token_log_mapping])

    # Create a DataFrame for the bar plot
    token_data = pd.DataFrame({
        "Tokens": tokens,
        "Log Probability": log_probs
    })

    # Create a bar plot for log probabilities
    fig = px.bar(
        token_data,
        x="Tokens",
        y="Log Probability",
        title="Log Probability per Token",
    )

    return token_details, fig
# Run the Dash app
if __name__ == "__main__":
    app.run_server(debug=True)
