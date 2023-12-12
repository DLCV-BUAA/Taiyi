import plotly.graph_objects as go


def Surface3d(data, title):
    z_data = []
    for v in data.values():
        z_data.append(v)
    x_data = list(range(1, len(z_data) + 1))
    y_data = list(range(1, len(z_data[0]) + 1))
    layout = go.Layout(
        title=title,
        scene=dict(
            xaxis=dict(title='step'),
            zaxis=dict(title='val')
        )
    )
    fig = go.Figure(
        data=[go.Surface(
            x=x_data,
            y=y_data,
            z=z_data,
        )],
        layout=layout
    )
    return fig


if __name__ == '__main__':
    a = {'0': [1, 2, 3], '1': [1, 2, 3], '2': [1, 2, 3]}
    fig = Surface3d(a)
    fig.show()
