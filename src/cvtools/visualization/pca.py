import numpy as np
import mpl_toolkits.mplot3d
import matplotlib.pyplot as plt
from sklearn.decomposition import IncrementalPCA


plt.style.use('seaborn-v0_8-notebook')

def pca_visualization(
        features,
        labels,
        class_names,
        n_components=2,
        batch_size=1000,
        figsize=(10, 8)
    ):

    pca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
    reduced_features = pca.fit_transform(features)

    print(f"Total variance explained: {np.sum(pca.explained_variance_ratio_)}")

    fig = plt.figure(1, figsize=figsize)

    if n_components == 2:
        ax = fig.add_subplot(111)
        scatter = ax.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels, cmap="tab20")

    elif n_components == 3:
        ax = fig.add_subplot(111, projection='3d', elev=-150, azim=110)

        scatter = ax.scatter(
            reduced_features[:, 0],
            reduced_features[:, 1],
            reduced_features[:, 2],
            c=labels,
            cmap="tab20"
        )
        ax.set_zlabel('PC 3')
        ax.zaxis.set_ticklabels([])

    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    ax.legend(
        scatter.legend_elements(num=len(class_names))[0],
        class_names,
        loc="center left",
        title="Classes",
        bbox_to_anchor=(1, 0.5),
        ncol=1,
    )

    plt.show()
