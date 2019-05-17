import numpy as np


def calc_side_lengths(triangles):
    """
    Calculate side lengths for a triangle. Outputs ascending side lengths.

    Inputs:
    triangles - (n, 3, 2) - 1st dimension is triangle number, 2nd dimension is
                            vertex number, 3rd dimension is coordinate

    outputs:
    ell - (n, 3) - sidelengths of triangle. Ascending order
                   (l_min, l_med, l_max)
    """
    N = triangles.shape[0]
    # Index to cyclic permute vertices
    ind = [2, 0, 1]
    shifted = triangles[:, ind]
    diff = triangles - shifted
    square = diff**2
    lengths = np.sqrt(np.sum(square, axis=2))
    return np.sort(lengths, axis=1)


def calc_area_of_triangle(l):
    """
    Calculate area of triangle

    Inputs:
    l - (3,) / (n, 3). lengths of triangle sides

    outputs:
    A - float. Area of triangle
    """
    l = np.atleast_2d(l)
    s = np.sum(l, axis=1) / 2
    s = np.atleast_2d(s).T
    coefs = s-l
    A = np.sqrt(s.T*np.product(coefs, axis=1))
    return np.squeeze(A)


def calc_min_angle_norm(A, l):
    """
    Calculate the normalized minimum angle of the triangle:
    theta_min = (3/pi) * arcsin(2A/(l_max*l_med))

    Inputs:
    A - (n,)   - Area of each triangle
    l - (n, 3) - length of each triangle, in ascending order

    outputs:
    theta - (n,) - Minimum angle
    """
    l = np.atleast_2d(l)
    l_used = l[:, 1:]
    theta = (3/np.pi) * np.arcsin(2*A/np.product(l_used, axis=1))
    return theta


def calc_aspect_ratio_norm(A, l):
    """
    Calculate the normalized aspect ratio of the triangles:
    r = 4*sqrt(3)*A/sum(l**2)

    Inputs:
    A - (n,)   - Area of each triangle
    l - (n, 3) - length of each triangle, in ascending order

    outputs:
    theta - (n,) - Minimum angle
    """
    l = np.atleast_2d(l)
    r = 4*np.sqrt(3)*A/np.sum(l**2, axis=1)
    return r


if __name__ == "__main__":
    a = np.array((((2, 2), (4, 1), (6, 3)), ((2, 9), (8, 11), (6, 5))))
    lengths = calc_side_lengths(a)
    areas = calc_area_of_triangle(lengths)
    theta = calc_min_angle_norm(areas, lengths)
    r = calc_aspect_ratio_norm(areas, lengths)
