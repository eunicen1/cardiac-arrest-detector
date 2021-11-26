import matplotlib.pyplot as plt

import numpy as np

def plot_classifier(model, X, y, need_argmax=False, ax=None):
    """plots the decision boundary of the model and the scatterpoints
       of the target values 'y'.

    Assumptions
    -----------
    y : it should contain two classes: '1' and '2'

    Parameters
    ----------
    model : the trained model which has the predict function

    X : the N by D feature array

    y : the N element vector corresponding to the target values

    """
    x1 = X[:, 0]
    x2 = X[:, 1]

    x1_min, x1_max = int(x1.min()) - 1, int(x1.max()) + 1
    x2_min, x2_max = int(x2.min()) - 1, int(x2.max()) + 1

    x1_line = np.linspace(x1_min, x1_max, 200)
    x2_line = np.linspace(x2_min, x2_max, 200)

    x1_mesh, x2_mesh = np.meshgrid(x1_line, x2_line)

    mesh_data = np.c_[x1_mesh.ravel(), x2_mesh.ravel()]

    y_pred = model.predict(mesh_data)
    if need_argmax:
        y_pred = np.argmax(y_pred, axis=1)

    y_pred = np.reshape(y_pred, x1_mesh.shape)

    if ax is None:
        ax = plt.gca()
    ax.set_xlim([x1_mesh.min(), x1_mesh.max()])
    ax.set_ylim([x2_mesh.min(), x2_mesh.max()])

    ax.contourf(
        x1_mesh,
        x2_mesh,
        -y_pred.astype(int),  # unsigned int causes problems with negative sign... o_O
        cmap=plt.cm.RdBu,
        alpha=0.6,
    )

    y_vals = np.unique(y)
    for c, color in zip(y_vals, "br"):
        in_c = y == c
        ax.scatter(x1[in_c], x2[in_c], color=color, label=f"class {c:+d}")
    ax.legend()

    plt.show()
