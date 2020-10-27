import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def _construct_df(dates, predicted, actual):
    df_dict = {
        'Dates': dates.to_numpy(),
        'Predicted': predicted.flatten(),
        'Actual': actual.flatten()
    }
    return pd.DataFrame(df_dict)


def pretty_plot(dates, predicted, actual, rmse):
    df = _construct_df(dates, predicted, actual)
    layout = go.Layout(
        title=dict(
            text=f'Predicted vs Actual Irradiance (RMSE: {rmse})', x=0.5
        ),
        font=dict(size=18),
        xaxis=dict(showgrid=False, title='Date'),
        yaxis=dict(showgrid=False, zeroline=False, title='GHI (W/m^2)'),
        colorway=px.colors.qualitative.Plotly
    )
    fig = go.Figure(layout=layout)
    fig.add_trace(
        go.Scatter(
            x=df['Dates'], y=df['Actual'], name='Actual', line=dict(width=3)
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df['Dates'],
            y=df['Predicted'],
            mode='lines',
            name='Predicted',
            line=dict(width=3)
        )
    )
    fig.show()
