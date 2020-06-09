import numpy as np


def proj(u, v):
    return u * np.dot(v, u) / np.dot(u, u)


def GS(V):
    V = 1.0 * V
    U = np.copy(V)
    for i in range(1, V.shape[1]):
        for j in range(i):
            U[:, i] -= proj(U[:, j], V[:, i])
    den = (U ** 2).sum(axis=0) ** 0.5
    E = U / den
    return E


def get_axis(CA_coord, C_coord, N_coord):
    vec_ac = np.array([CA_coord[0] - C_coord[0], CA_coord[1] - C_coord[1], CA_coord[2] - C_coord[2]])
    vec_an = np.array([CA_coord[0] - N_coord[0], CA_coord[1] - N_coord[1], CA_coord[2] - N_coord[2]])
    vec_ab = np.cross(vec_an, vec_ac)
    vec_ac, vec_an, vec_ab = GS(np.array([vec_ac, vec_an, vec_ab]))
    axis = np.array([vec_ac, vec_an, vec_ab])
    return axis
