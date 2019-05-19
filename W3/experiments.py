from spring_method import *


def ex1():
    np.random.seed(42)
    X, Y, simplices, im = create_mesh('example.bmp', N_verts=500, N_iter=15,
                                      include_self=True, N_points=10, m_max=1)

    fig, (ax1, ax2) = plt.subplots(figsize=(9, 3.2), ncols=2)
    ax1.imshow(im, cmap='Greys_r')
    ax1.triplot(X, Y, simplices, color='b')
    ax1.scatter(X, Y, color='b', s=5)
    fig.suptitle("$N_{vertices} = 500, N_{iterations} = 15, N_{points} = 10$")
    xborder = 35
    yborder = 50
    ax1.set_xlim(xborder, im.shape[0]-1-xborder)
    ax1.set_ylim(yborder, im.shape[1]-1-yborder)

    ax2 = plot_quality(simplices, X, Y, ax=ax2)
    fig.tight_layout()
    fig.savefig('ex1.pdf')


def ex2():
    np.random.seed(42)
    X, Y, simplices, im = create_mesh('example.bmp', N_verts=500, N_iter=15,
                                      include_self=True, N_points=10, m_max=10)

    fig, (ax1, ax2) = plt.subplots(figsize=(9, 3.2), ncols=2)
    ax1.imshow(im, cmap='Greys_r')
    ax1.triplot(X, Y, simplices, color='b')
    ax1.scatter(X, Y, color='b', s=5)
    fig.suptitle(
        "$N_{vertices} = 500, N_{iterations} = 15, N_{points} = 10, m_{max}=10$")
    xborder = 35
    yborder = 50
    ax1.set_xlim(xborder, im.shape[0]-1-xborder)
    ax1.set_ylim(yborder, im.shape[1]-1-yborder)

    ax2 = plot_quality(simplices, X, Y, ax=ax2)
    fig.tight_layout()
    fig.savefig('ex2.pdf')

def ex3():
    x, y, simplices = read_from_triangle('EG_200.1')
    Gx, Gy, sdf, X, Y, im, sdf_spline = import_data('EG_WEB_logo.jpg',
                                                    invert=True)
    fig, (ax1, ax2) = plt.subplots(figsize=(8, 4), ncols=2)
    ax1.imshow(im, cmap='Greys_r')
    ax1.triplot(x, y, simplices, color='b')
    ax1.scatter(x, y, color='b', s=5)
    # xborder = 35
    # yborder = 50
    # ax1.set_xlim(xborder, im.shape[0]-1-xborder)
    # ax1.set_ylim(yborder, im.shape[1]-1-yborder)

    ax2 = plot_quality(simplices, x, y, ax=ax2)
    fig.suptitle("$N_{perim} = 200, A_{max} = 50$")
    fig.tight_layout()
    fig.savefig('ex3.pdf')


def ex4():
    np.random.seed(42)
    X, Y, simplices, im = create_mesh('EG_WEB_logo.jpg', N_verts=500, N_iter=15,
                                      include_self=True, N_points=10, m_max=20,
                                      invert=True)

    fig, (ax1, ax2) = plt.subplots(figsize=(8, 4), ncols=2)
    ax1.imshow(im, cmap='Greys_r')
    ax1.triplot(X, Y, simplices, color='b')
    ax1.scatter(X, Y, color='b', s=5)
    fig.suptitle(
        "$N_{vertices} = 500, N_{iterations} = 15, N_{points} = 10, m_{max}=20$")
    ax2 = plot_quality(simplices, X, Y, ax=ax2)
    fig.tight_layout()
    fig.savefig('ex4.pdf')


def ex5():
    x, y, simplices = read_from_triangle('bmp_ex.1')
    Gx, Gy, sdf, X, Y, im, sdf_spline = import_data('example.bmp')
    fig, (ax1, ax2) = plt.subplots(figsize=(8, 4), ncols=2)
    ax1.imshow(im, cmap='Greys_r')
    ax1.triplot(x, y, simplices, color='b')
    ax1.scatter(x, y, color='b', s=5)
    # xborder = 35
    # yborder = 50
    # ax1.set_xlim(xborder, im.shape[0]-1-xborder)
    # ax1.set_ylim(yborder, im.shape[1]-1-yborder)

    ax2 = plot_quality(simplices, x, y, ax=ax2)
    fig.suptitle("$N_{perim} = 100, A_{max} = 100$")
    fig.tight_layout()
    fig.savefig('ex5.pdf')


def ex6():
    x, y, simplices = read_from_triangle('islands.1')
    Gx, Gy, sdf, X, Y, im, sdf_spline = import_data('test.png')
    fig, (ax1, ax2) = plt.subplots(figsize=(8, 4), ncols=2)
    ax1.imshow(im, cmap='Greys_r')
    ax1.triplot(x, y, simplices, color='b')
    ax1.scatter(x, y, color='b', s=5)
    # xborder = 35
    # yborder = 50
    # ax1.set_xlim(xborder, im.shape[0]-1-xborder)
    # ax1.set_ylim(yborder, im.shape[1]-1-yborder)

    ax2 = plot_quality(simplices, x, y, ax=ax2)
    fig.suptitle("$N_{perim} = 150, A_{max} = 100$")
    fig.tight_layout()
    fig.savefig('ex6.pdf')


def ex7():
    np.random.seed(42)
    X, Y, simplices, im = create_mesh('test.png', N_verts=1000, N_iter=15,
                                      include_self=True, N_points=10, m_max=20,
                                      invert=False)

    fig, (ax1, ax2) = plt.subplots(figsize=(8, 4), ncols=2)
    ax1.imshow(im, cmap='Greys_r')
    ax1.triplot(X, Y, simplices, color='b')
    ax1.scatter(X, Y, color='b', s=5)
    fig.suptitle(
        "$N_{vertices} = 1000, N_{iterations} = 15, N_{points} = 10, m_{max}=20$")
    ax2 = plot_quality(simplices, X, Y, ax=ax2)
    fig.tight_layout()
    fig.savefig('ex7.pdf')
