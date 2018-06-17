
from datetime import datetime
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import svds
import pandas as pd
import numpy as np

"""Load data"""
data = pd.read_excel("zurich_insurance.xlsx")
data.head() # first 5 lines
data.tail() # show only the last 5 lines

"""Preprocessing"""
# Obtaining a Complete Dataset (Dropping Missing Values)
data = data.dropna()
X = data.reset_index(drop = True)

pre = 7
service_matrix = X.as_matrix()[:,pre:17].astype(float)


def recommend(adds, service_matrix=service_matrix, k=3):
    if adds:
        service_matrix = np.insert(service_matrix, service_matrix.shape[0], adds, axis=0)
    #get SVD components from train matrix. Choose k.
    u, s, vt = svds(service_matrix, k = k)
    s_diag_matrix=np.diag(s)
    X_pred = np.dot(np.dot(u, s_diag_matrix), vt)
    # print('User-based CF MSE: ' + str(rmse(X_pred, service_matrix)))
    return X_pred, service_matrix


from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go


new_client = [1, 0, 0, 0, 1, 0, 0, 0, 0, 0]
new_client = [1, 0.5, 1, 0.5, 0.5, 1, 0.5, 0.5, 0.5, 0.5]



X_pred, new_mat = recommend(new_client)


ins = X.columns[pre:17]
idx = range(len(ins))

client = -1
recomended = X_pred[client,:]
base = new_mat[client,:]

recomended *= base>0
recomended *= base<1
base[base==.5] = 0

print(" Most recommended package %s " % ins[np.argmax(recomended)])

trace1 = go.Bar(
    x=ins,
    y=base,
    name='Actual products',
    marker=dict(
        color='rgba(204,204,204,1)'
    )
)

trace2 = go.Bar(
    x=ins,
    y=recomended,
    name='Recomended products',
    marker=dict(
        color='rgba(222,45,38,0.8)'
    )
)

data = [trace1, trace2]


layout = go.Layout(
    title='Recomended policies for client',
    xaxis=dict(
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='',
        titlefont=dict(
            size=16,
            color='rgba(100, 100, 100, 1)',
        ),
        tickfont=dict(
            size=14,
            color='rgba(100, 100, 100, 1)',
        )
    ),
    legend=dict(
        x=.85,
        y=1.2,
        bgcolor='rgba(255, 255, 255, 0)',
        bordercolor='rgba(255, 255, 255, 0)'
    ),
    barmode='stack',
    bargap=0.15,
    bargroupgap=0.1
)

fig = go.Figure(data=data, layout=layout)
plot(fig, filename='style-bar')
# init_notebook_mode(connected=True)
