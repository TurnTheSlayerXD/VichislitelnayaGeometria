

import numpy as np
import matplotlib.pyplot as plt


def display_polys(polygons: np.ndarray):
    
    plt.figure()
    colors = np.array(['red', 'green', 'blue', 'yellow'])

    max_x = np.abs(polygons[:,:, 0]).max()
    max_y = np.abs(polygons[:,:, 1]).max()
    
    fin = np.max([max_x, max_y])
    
    plt.xlim(-fin, fin)
    plt.ylim(-fin, fin)

    n_figs = polygons.shape[0]
    n_vertices = polygons.shape[1]

    plt.grid()
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')

    fin_colors = [] 
    for color in colors[:n_figs]:
        fin_colors += list(np.repeat(color, n_vertices))

    # plt.scatter(cords[:, 0], cords[:, 1], s=170,
    #             color=fin_colors)
    for i, poly in enumerate(polygons):
        t = plt.Polygon(poly[:, (0, 1)], alpha=0.5, color=colors[i])
        plt.gca().add_patch(t)
        print(t)

    plt.show()


def display_points(points: list, labels=['unknown']):
    
    colors = np.array(['black', 'red', 'orange', 'yellow', 'green', 'blue', 'violet'])
  
    plt.grid()

    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')

    for i, line in enumerate(points):
        label=labels[i % len(labels)]
        if label != 'unknown':
            plt.plot(line[:, 0], line[:, 1], 'o', color=colors[i % len(colors)], markersize=2,
                    label=labels[i % len(labels)])
        else:
            plt.plot(line[:, 0], line[:, 1], 'o', color=colors[i % len(colors)], markersize=2)

            
    plt.legend(loc='best')

    plt.show()
