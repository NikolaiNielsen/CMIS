import numpy as np
import matplotlib.pyplot as plt
import sys
import scipy.io as io
from progress.bar import Bar
sys.path.append('../useful_functions')
import useful_functions as uf
import mesh
import fem

np.set_printoptions(threshold=np.inf)


def D(E=69e9, nu=0.3):
    """
    Returns the Elasticity matrix "D" for a given Young Modulus "E" and Poisson
    ratio "nu".  Defaults are for steel.
    """
    return E/(1-nu*nu) * np.array([[1, nu, 0],[nu, 1, 0],[0, 0, (1-nu)/2]])


STEEL_D = D()


def create_B(triangle, area):
    """
    Create the B-matrix for a given element, given the element positions:
    triangle: (3, 2) matrix
    B = S @ N
    S = [[dx, 0],
         [0, dy],
         [dy, dx]]
    N = [[N^i_x, 0,     N^j_x, 0,     N^k_x, 0    ],
         [0,     N^i_y, 0,     N^j_y,     0, N^k_y]]
    """
    diff_vectors = triangle[[2, 0, 1]] - triangle[[1, 2, 0]]
    dx = -diff_vectors[:, 1]
    dy = diff_vectors[:, 0]
    B = (1/2/area) * np.array([[dx[0],     0, dx[1],     0, dx[2],     0],
                               [0,     dy[0],     0, dy[1],     0, dy[2]],
                               [dy[0], dx[0], dy[1], dx[1], dy[2], dx[2]]])
    return B


def create_element_matrix(triangle, area, D=STEEL_D):
    B = create_B(triangle, area)
    Ke = area * B.T @ D @ B
    return Ke


def load_mat(file, return_areas=False):
    mat = io.loadmat(file)
    x = mat['X'].flatten()
    y = mat['Y'].flatten()
    simplices = mat['T'] - 1
    if return_areas:
        areas = mat['A']
        return x, y, simplices, areas
    return x, y, simplices


def func_source(tri):
    return 0


def calc_hat_area(x):
    """
    Calculates the sum of integrals of hat functions for a list of points:
    The integral is just 1/2 l_e, where l_e is the length of the element. But
    inner nodes have contributions from two elements, so we add these twice

    Inputs:
    - x, (n,) array of positions

    Returns:
    - A_n (n,) area per node
    """
    # Get the permutations to sort and unsort x, just incase it isn't sorted
    perm = np.argsort(x)
    inv_perm = np.arange(perm.size)[np.argsort(perm)]

    # sort x
    x = x[perm]
    le = x[1:] - x[:-1]
    A_n = np.zeros(x.shape)
    A_n[1:] += le
    A_n[:-1] += le

    # unsort and return A_n
    A_n = A_n[inv_perm]
    return A_n / 2


def ex_with_external():
    x, y, simplices = load_mat('data.mat')

    # Clamp the left side
    mask1 = x == np.amin(x)
    vals1 = 0

    # Get the element matric keyword arguments
    elem_dict = dict(D=STEEL_D)

    # We apply a downwards nodal force on the right side
    bottom_force = -5e7
    mask2 = x == np.amax(x)
    A_n = calc_hat_area(y[mask2])
    vals2 = np.zeros((A_n.size, 2))
    vals2[:,1] =A_n * bottom_force

    x_displacement, y_displacement = fem.FEM(x, y, simplices,
                                             create_element_matrix,
                                             func_source,
                                             mask1, vals1, mask2, vals2,
                                             elem_dict=elem_dict)
    x_new = x + x_displacement
    y_new = y + y_displacement
    fig, ax = plt.subplots()
    ax.triplot(x, y, simplices, color='r')
    ax.triplot(x_new, y_new, simplices, color='b')

    triangles_before = mesh.all_triangles(simplices, x, y)
    triangles_after = mesh.all_triangles(simplices, x_new, y_new)
    area_before = np.sum(fem.calc_areas(triangles_before))
    area_after = np.sum(fem.calc_areas(triangles_after))
    print(area_before)
    print(area_after)
    plt.show()


def ex_resolution(max_max_area=2, min_max_area=0.01, num=50):
    contour = [np.array([[0,0],[6,0],[6,2],[0,2]])]
    end_area = np.zeros(num)
    end_positions = np.zeros((num, 2))
    bottom_traction = -5e7
    max_areas = np.logspace(np.log2(max_max_area), np.log2(min_max_area),
                            num=num, base=2)
    savefile = 'areas'
    savefile2 = 'positions'
    bar = Bar('Simulate', max=num)
    for i, area in enumerate(max_areas):
        x, y, simplices = mesh.generate_and_import_mesh(contour,
                                                        max_area=area)
        bottom_right_index = (x==np.amax(x)) * (y==np.amin(y))
        mask1 = x == np.amin(x)
        vals1 = 0

        # Get the element matric keyword arguments
        elem_dict = dict(D=STEEL_D)

        # We apply a downwards nodal force on the right side
        mask2 = x == np.amax(x)
        A_n = calc_hat_area(y[mask2])
        vals2 = np.zeros((A_n.size, 2))
        vals2[:, 1] = A_n * bottom_traction

        x_displacement, y_displacement = fem.FEM(x, y, simplices,
                                                create_element_matrix,
                                                func_source,
                                                mask1, vals1, mask2, vals2,
                                                elem_dict=elem_dict)
        x_new = x + x_displacement
        y_new = y + y_displacement
        triangles_after = mesh.all_triangles(simplices, x_new, y_new)
        end_area[i] = np.sum(fem.calc_areas(triangles_after))
        end_positions[i,:] = [x_new[bottom_right_index],
                              y_new[bottom_right_index]]
        np.save(savefile, end_area)
        np.save(savefile2, end_positions)
        bar.next()
    bar.finish()


def ex_resolution_raw(max_max_area=2, min_max_area=0.01):
    elem_dict = dict(D=STEEL_D)
    contour = [np.array([[0, 0], [6, 0], [6, 2], [0, 2]])]
    bottom_traction = -5e7
    x1, y1, simplices1 = mesh.generate_and_import_mesh(contour,
                                                       max_area=max_max_area)
    mask1 = x1 == np.amin(x1)
    vals1 = 0

    # We apply a downwards nodal force on the right side
    mask2 = x1 == np.amax(x1)
    A_n = calc_hat_area(y1[mask2])
    vals2 = np.zeros((A_n.size, 2))
    vals2[:, 1] = A_n * bottom_traction

    x_displacement, y_displacement = fem.FEM(x1, y1, simplices1,
                                             create_element_matrix,
                                             func_source,
                                             mask1, vals1, mask2, vals2,
                                             elem_dict=elem_dict)
    x1_new = x1 + x_displacement
    y1_new = y1 + y_displacement

    np.save('raw_exps1', [x1, y1, x1_new, y1_new, simplices1])


def ex_analyze_resolution(savefile='areas.npy', savefile2='positions.npy'):
    start_areas = 12
    areas = np.load(savefile)
    res_areas = areas - start_areas
    positions = np.load(savefile2)
    max_areas = np.load('x.npy')
    res_x = positions[:,0] - 6
    res_y = positions[:,1] - 0
    [x1, y1, x1_new, y1_new, simplices1] = np.load('raw_exps1.npy')
    [x2, y2, x2_new, y2_new, simplices2] = np.load('raw_exps2.npy')
    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(5.5, 3.5))
    ax1, ax2, ax3, ax4 = ax.flatten()
    ax1.plot(max_areas, res_areas)
    ax1.set_title('Growth in area')
    ax1.set_xlabel('Max $A^e$')
    ax1.set_ylabel('$\Delta A$ [m$^2$]')
    ax1.set_xscale('log')

    ax2.plot(max_areas, res_x, label='$\Delta x$')
    ax2.plot(max_areas, res_y, label='$\Delta y$')
    ax2.set_title('Displacement for lower right node')
    ax2.set_xscale('log')
    ax2.set_xlabel('Max $A^e$')
    ax2.set_ylabel('$u$ [m]')
    ax2.legend()

    ax3.triplot(x1, y1, simplices1)
    ax3.triplot(x1_new, y1_new, simplices1)
    ax3.set_title(r'Max $A^e$: 2 m$^2$')
    ax3.set_xlabel('x, [m]')
    ax3.set_ylabel('y, [m]')
    ax3.set_aspect('equal')

    ax4.triplot(x2, y2, simplices2, linewidth=0.5)
    ax4.triplot(x2_new, y2_new, simplices2, linewidth=0.5)
    ax4.set_title('Max $A^e$: 0.001 m$^2$')
    ax4.set_xlabel('x, [m]')
    ax4.set_ylabel('y, [m]')
    ax4.set_aspect('equal')

    fig.tight_layout()
    fig.savefig('handin/ex_resolution.pdf')


def ex_with_external2():
    contour = [np.array([[0, 0], [6, 0], [6, 2], [0, 2]])]
    bottom_traction = -5e8
    x, y, simplices = mesh.generate_and_import_mesh(contour,
                                                    max_area=0.1)

    # Clamp the left side
    mask1 = x == np.amin(x)
    vals1 = 0

    # Get the element matric keyword arguments
    elem_dict = dict(D=STEEL_D)

    # We apply a downwards nodal force on the right side
    mask2 = x == np.amax(x)
    A_n = calc_hat_area(y[mask2])
    vals2 = np.zeros((A_n.size, 2))
    vals2[:, 1] = A_n * bottom_traction

    x_displacement, y_displacement = fem.FEM(x, y, simplices,
                                             create_element_matrix,
                                             func_source,
                                             mask1, vals1, mask2, vals2,
                                             elem_dict=elem_dict)
    x_new = x + x_displacement
    y_new = y + y_displacement
    fig, ax = plt.subplots(figsize=(4,3))
    ax.triplot(x, y, simplices, color='r')
    ax.triplot(x_new, y_new, simplices, color='b')
    ax.set_title(f'Linear deformation of steel. $t=5\cdot 10^8$ N/m')
    ax.set_aspect('equal')
    ax.set_xlabel('x, [m]')
    ax.set_ylabel('y, [m]')
    fig.tight_layout()
    # triangles_before = mesh.all_triangles(simplices, x, y)
    # triangles_after = mesh.all_triangles(simplices, x_new, y_new)
    # area_before = np.sum(fem.calc_areas(triangles_before))
    # area_after = np.sum(fem.calc_areas(triangles_after))
    # print(area_before)
    # print(area_after)
    # np.save('raw_exps2', [x, y, x_new, y_new, simplices])
    fig.savefig('handin/raw.pdf')


def ex_load(min_load=0, max_load=-5e7, num=100, max_area=0.01):
    contour = [np.array([[0, 0], [6, 0], [6, 2], [0, 2]])]
    x, y, simplices = mesh.generate_and_import_mesh(contour,
                                                    max_area=max_area)

    start_area = 12
    savefile='loads'
    areas = np.zeros(num)
    loads = np.linspace(min_load, max_load, num=num)

    mask1 = x == np.amin(x)
    vals1 = 0

    # Get the element matric keyword arguments
    elem_dict = dict(D=STEEL_D)

    # We apply a downwards nodal force on the right side
    mask2 = x == np.amax(x)
    A_n = calc_hat_area(y[mask2])
    vals2 = np.zeros((A_n.size, 2))
    vals2[:, 1] = A_n

    K = fem.assemble_global_matrix(x, y, simplices, create_element_matrix,
                                   elem_dict=elem_dict)
    f = fem.create_f_vector(x, y, simplices, func_source)
    K, f = fem.add_point_boundary(K, f, mask1, vals1)
    # bar = Bar('Simulating', max=num)
    for i, load in enumerate(loads):
        vals = vals2*load
        f = fem.create_f_vector(x, y, simplices, func_source)
        f = fem.add_to_source(f, mask2, vals)
        u = np.linalg.solve(K, f)
        x_disp, y_disp = fem.unpack_u(u)
        x_new = x + x_disp
        y_new = y + y_disp
        triangles = mesh.all_triangles(simplices, x_new, y_new)
        area = np.sum(fem.calc_areas(triangles))
        areas[i] = area
        np.save(savefile, [loads, areas])
        # bar.next()
    # bar.finish()


def ex_analyze_loads():
    loads, areas = np.load('loads.npy')
    start_area = 12
    res = areas - start_area
    # print(res)
    ratio = res/start_area
    for load, rat in zip(loads, ratio):
        print(load, rat)
    i = 10
    exponent = (np.log(ratio[-1])-np.log(ratio[i]))/(np.log(-loads[-1])-np.log(-loads[i]))
    # print(exponent)
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(-loads, ratio)
    # ax.scatter(loads[[i, -1]], areas[])
    ax.set_xlabel('Traction [N/m]')
    ax.set_ylabel('Relative area increase')
    # ax.set_yscale('log')
    # ax.set_xscale('log')
    ax.set_title('Relative area increase as function of traction')
    fig.tight_layout()
    fig.savefig('handin/ex_loads.pdf')


def ex_show_loads():
    contour = [np.array([[0, 0], [6, 0], [6, 2], [0, 2]])]
    x, y, simplices = mesh.generate_and_import_mesh(contour,
                                                    max_area=0.01)


    load = -5e7

    mask1 = x == np.amin(x)
    vals1 = 0

    # Get the element matric keyword arguments
    elem_dict = dict(D=STEEL_D)

    # We apply a downwards nodal force on the right side
    mask2 = x == np.amax(x)
    A_n = calc_hat_area(y[mask2])
    vals2 = np.zeros((A_n.size, 2))
    vals2[:, 1] = A_n

    K = fem.assemble_global_matrix(x, y, simplices, create_element_matrix,
                                   elem_dict=elem_dict)
    f = fem.create_f_vector(x, y, simplices, func_source)
    K, f = fem.add_point_boundary(K, f, mask1, vals1)
    vals = vals2*load
    f = fem.add_to_source(f, mask2, vals)
    u = np.linalg.solve(K, f)
    x_disp, y_disp = fem.unpack_u(u)
    x_new = x + x_disp
    y_new = y + y_disp
    fig, ax = plt.subplots()
    ax.triplot(x,y,simplices)
    ax.triplot(x_new, y_new, simplices)
    triangles = mesh.all_triangles(simplices, x_new, y_new)
    area = np.sum(fem.calc_areas(triangles))
    print(area)
    plt.show()


def ex_squares():
    poissons = [0.1, 0.3, 0.5]
    max_areas = [0.5, 0.1, 0.01]
    contour = [np.array([[0, 0], [6, 0], [6, 2], [0, 2]])]
    save_file = 'squares'
    poissons2, max_areas2 = np.meshgrid(poissons, max_areas)

    num = 100
    min_load = 0
    max_load = -5e7
    areas = np.zeros((num, *poissons2.shape))
    loads = np.linspace(min_load, max_load, num=num)
    bar = Bar('simulating', max=areas.size)
    for i in range(poissons2.shape[0]):
        for j in range(poissons2.shape[1]):
            x, y, simplices = mesh.generate_and_import_mesh(
                contour, max_area=max_areas2[i,j])
            elem_dict = dict(D=D(nu=poissons2[i,j]))
            mask1 = x == np.amin(x)
            vals1 = 0
            # We apply a downwards nodal force on the right side
            mask2 = x == np.amax(x)
            A_n = calc_hat_area(y[mask2])
            vals2 = np.zeros((A_n.size, 2))
            vals2[:, 1] = A_n

            K = fem.assemble_global_matrix(x, y, simplices,
                                           create_element_matrix,
                                           elem_dict=elem_dict)
            f = fem.create_f_vector(x, y, simplices, func_source)
            K, f = fem.add_point_boundary(K, f, mask1, vals1)
            # bar = Bar('Simulating', max=num)
            for n, load in enumerate(loads):
                vals = vals2*load
                f = fem.create_f_vector(x, y, simplices, func_source)
                f = fem.add_to_source(f, mask2, vals)
                u = np.linalg.solve(K, f)
                x_disp, y_disp = fem.unpack_u(u)
                x_new = x + x_disp
                y_new = y + y_disp
                triangles = mesh.all_triangles(simplices, x_new, y_new)
                area = np.sum(fem.calc_areas(triangles))
                areas[n,i,j] = area
                np.save(save_file, areas)
                bar.next()
    bar.finish()


def ex_analyze_squares():

    poissons = [0.1, 0.3, 0.5]
    max_areas = [1, 0.1, 0.01]
    loads = np.linspace(0, 5e7, num=100)
    poissons2, max_areas2 = np.meshgrid(poissons, max_areas)
    areas = np.load('squares.npy')
    start_area = 12
    res = areas-start_area
    ratio = res/start_area
    n = 20
    
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(9, 9))

    for i in range(poissons2.shape[0]):
        for j in range(poissons2.shape[1]):
            exponent = (np.log(ratio[-1, i, j])-np.log(ratio[n, i, j])) / \
                (np.log(loads[-1])-np.log(loads[n]))
            print(exponent)
            axes[i,j].plot(loads, ratio[:, i, j])
            axes[i,j].scatter(loads[[n, -1]], ratio[[n, -1], i, j])
            axes[i,j].set_xlabel('Traction [N/m]')
            axes[i,j].set_ylabel('Relative area increase')
            axes[i,j].set_yscale('log')
            axes[i,j].set_xscale('log')
            axes[i,j].set_title(
                r'$\nu$ = ' + str(poissons2[i,j]) + ', $A^e$ = ' + str(max_areas2[i,j]) + r', $\Delta A / \Delta t$ = ' + f'{exponent:.3f}')
    
    fig.tight_layout()
    fig.savefig('handin/squares.pdf')


ex_analyze_loads()