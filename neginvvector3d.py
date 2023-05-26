import torch
import math
from math import pi, sqrt
from functools import lru_cache, wraps


def nan_to_zero(func):
    """ Decorator that will replace all nans with zeros in the returned tensor. """
    @wraps(func)
    def wrapper(*args, **kwargs):
        return torch.nan_to_num(func(*args, **kwargs))
    return wrapper


def _sign(x: torch.Tensor):
    """ Returns values of 1 or -1. When x == 0, the output depends on the type of x. """
    return torch.copysign(torch.ones_like(x), x)


@nan_to_zero
def vector_2_doubleangle(v: torch.Tensor):
    """
    Doubleangle representation in 2D.
    v has shape (batch, 2, ...)
    returns a tensor of shape (batch, 2, ...)
    """
    x, y = v[:, 0:1, ...], v[:, 1:2, ...]
    x2, y2 = x**2, y**2
    doubleangle_0 = (x2 - y2) / (x2 + y2)
    doubleangle_1 = (2*x*y) / (x2 + y2)
    return torch.cat([doubleangle_0, doubleangle_1], dim=1)


@nan_to_zero
def doubleangle_2_vector(r: torch.Tensor):
    """
    Doubleangle representation in 2D.
    r has shape (batch, 2, ...)
    returns a tensor with shape (batch, 2, ...)
    """
    r0 = r[:, 0:1, ...]
    r1 = r[:, 1:2, ...]
    magnitude = torch.norm(r, dim=1, keepdim=True)
    x = torch.sqrt(0.5 + r0 / (2 * magnitude))
    y = _sign(r1) * torch.sqrt(0.5 - r0 / (2 * magnitude))
    return torch.cat([x, y], dim=1)


def vector_align(v: torch.Tensor, dim: int):
    """
    Multiplies vectors by 1 or -1 such that the dimension dim of the vector is positive.
    For the Z-Aligned-Vector representation, use dim=2.
    v has shape (batch, 3, ...)
    """
    return v * _sign(v[:, dim:dim + 1, ...])


@nan_to_zero
def vector_2_dip90_strike360(v: torch.Tensor):
    """
    Dip 90 - Strike 360 representation
    v has shape (batch, 3, ...)
    returns a tensor of shape (batch, 3, ...)
    """
    v_normalised = v / torch.norm(v, dim=1, keepdim=True)
    v_aligned = vector_align(v_normalised, 2)

    dip = torch.acos(v_aligned[:, 2:3, ...])
    dip_normalised = dip * 4 / pi - 1  # in [-1, 1]

    # strike = torch.atan2(-v_aligned[:, 1:2, ...], v_aligned[:, 0:1, ...])
    # cos_strike = torch.cos(strike)
    # sin_strike = torch.sin(strike)

    v_projection = v_aligned[:, 0:2, ...] / torch.norm(v_aligned[:, 0:2, ...], dim=1, keepdim=True)  # unit vector on x y plane
    cos_strike = v_projection[:, 0:1, ...]
    sin_strike = -v_projection[:, 1:2, ...]

    return torch.cat([dip_normalised, cos_strike, sin_strike], dim=1)


@nan_to_zero
def dip90_strike360_2_vector(r: torch.Tensor):
    """
    Dip 90 - Strike 360 representation
    r has shape (batch, 3, ...)
    returns a tensor of shape (batch, 3, ...)
    """
    dip_normalised = r[:, 0:1, ...]
    dip = (dip_normalised + 1) * pi / 4

    # strike = torch.atan2(r[:, 2:3, ...], r[:, 1:2, ...])
    # cos_strike = torch.cos(strike)
    # sin_strike = torch.sin(strike)

    cos_strike = r[:, 1:2, ...]
    sin_strike = r[:, 2:3, ...]

    z = torch.cos(dip)
    sin_dip = torch.sin(dip)
    x = cos_strike * sin_dip
    y = -sin_strike * sin_dip
    v = torch.cat([x, y, z], dim=1)

    v = v / torch.norm(v, dim=1, keepdim=True)
    return v


@nan_to_zero
def vector_2_dip180_strike180(v: torch.Tensor):
    """
    Dip 180 - Strike 180 representation
    v has shape (batch, 3, ...)
    returns a tensor of shape (batch, 3, ...)
    """
    v_normalised = v / torch.norm(v, dim=1, keepdim=True)
    v_aligned = vector_align(v_normalised, 0)

    dip = torch.acos(v_aligned[:, 2:3, ...])
    dip_normalised = dip * 2 / pi - 1  # in [-1, 1]

    # strike = torch.atan2(-v_aligned[:, 1:2, ...], v_aligned[:, 0:1, ...])
    # strike_doubleangle_0 = torch.cos(2*strike)
    # strike_doubleangle_1 = torch.sin(2*strike)

    v_projection = v_aligned[:, 0:2, ...] / torch.norm(v_aligned[:, 0:2, ...], dim=1, keepdim=True)  # unit vector on x y plane
    # Strike angle of the normal vector is measured clockwise from the x-axis.
    # Doubleangle is anticlockwise from the x-axis.
    # We need to reverse the y-axis component.
    v_projection_shifted = torch.cat([v_projection[:, 0:1, ...], -v_projection[:, 1:2, ...]], dim=1)
    strike_doubleangle = vector_2_doubleangle(v_projection_shifted)

    return torch.cat([dip_normalised, strike_doubleangle], dim=1)


@nan_to_zero
def dip180_strike180_2_vector(r: torch.Tensor):
    """
    Dip 180 - Strike 180 representation
    r has shape (batch, 3, ...)
    returns a tensor of shape (batch, 3, ...)
    """
    dip_normalised = r[:, 0:1, ...]
    dip = (dip_normalised + 1) * pi / 2

    # strike = 0.5 * torch.atan2(r[:, 2:3, ...], r[:, 1:2, ...])
    # cos_strike = torch.cos(strike)
    # sin_strike = torch.sin(strike)

    v_projection_shifted = doubleangle_2_vector(r[:, 1:3, ...])
    v_projection_shifted = vector_align(v_projection_shifted, dim=0)
    # Strike angle of the normal vector is measured clockwise from the x-axis
    # Doubleangle is anticlockwise from the x-axis.
    # We need to reverse the y-axis component.
    cos_strike = v_projection_shifted[:, 0:1, ...]
    sin_strike = v_projection_shifted[:, 1:2, ...]

    z = torch.cos(dip)
    sin_dip = torch.sin(dip)
    x = cos_strike * sin_dip
    y = -sin_strike * sin_dip
    v = torch.cat([x, y, z], dim=1)

    v = v / torch.norm(v, dim=1, keepdim=True)
    return v


@nan_to_zero
def vector_2_projection_doubleangle(v: torch.Tensor):
    """
    Projection-Doubleangle representation
    v has shape (batch, 3, ...)
    returns a tensor of shape (batch, 6, ...)
    """
    v_normalised = v / torch.norm(v, dim=1, keepdim=True)
    v_yz = v_normalised[:, (1, 2), ...]
    v_xz = v_normalised[:, (0, 2), ...]
    v_xy = v_normalised[:, (0, 1), ...]

    r_yz = vector_2_doubleangle(v_yz) * torch.norm(v_yz, dim=1, keepdim=True)
    r_xz = vector_2_doubleangle(v_xz) * torch.norm(v_xz, dim=1, keepdim=True)
    r_xy = vector_2_doubleangle(v_xy) * torch.norm(v_xy, dim=1, keepdim=True)

    return torch.cat([r_yz, r_xz, r_xy], dim=1)


@nan_to_zero
def projection_doubleangle_2_vector(r: torch.Tensor):
    """
    Projection-Doubleangle representation
    r has shape (batch, 6, ...)
    returns a tensor of shape (batch, 3, ...)
    """
    r_yz = r[:, 0:2, ...]
    r_xz = r[:, 2:4, ...]
    r_xy = r[:, 4:6, ...]

    v_yz = doubleangle_2_vector(r_yz)
    v_xz = doubleangle_2_vector(r_xz)
    v_xy = doubleangle_2_vector(r_xy)

    m_yz = torch.norm(r_yz, dim=1, keepdim=True)
    m_xz = torch.norm(r_xz, dim=1, keepdim=True)
    m_xy = torch.norm(r_xy, dim=1, keepdim=True)

    magnitude_x = torch.abs(v_xz[:, 0:1, ...]) + torch.abs(v_xy[:, 0:1, ...])
    magnitude_y = torch.abs(v_yz[:, 0:1, ...]) + torch.abs(v_xy[:, 1:2, ...])
    magnitude_z = torch.abs(v_yz[:, 1:2, ...]) + torch.abs(v_xz[:, 1:2, ...])
    smallest_x = (magnitude_x <= magnitude_y) & (magnitude_x <= magnitude_z)
    smallest_y = (magnitude_y < magnitude_x) & (magnitude_y <= magnitude_z)
    smallest_z = (magnitude_z < magnitude_x) & (magnitude_z < magnitude_y)

    sign_x_xz = _sign(v_xz[:, 0:1, ...])
    sign_x_xy = _sign(v_xy[:, 0:1, ...])
    sign_y_yz = _sign(v_yz[:, 0:1, ...])
    sign_y_xy = _sign(v_xy[:, 1:2, ...])
    sign_z_xz = _sign(v_xz[:, 1:2, ...])
    sign_z_yz = _sign(v_yz[:, 1:2, ...])

    s_yz = smallest_x + smallest_y * (sign_z_yz == sign_z_xz) + smallest_z * (sign_y_yz == sign_y_xy)
    s_xz = smallest_x * (sign_z_xz == sign_z_yz) + smallest_y + smallest_z * (sign_x_xz == sign_x_xy)
    s_xy = smallest_x * (sign_y_xy == sign_y_yz) + smallest_y * (sign_x_xy == sign_x_xz) + smallest_z
    s_yz = s_yz * 2 - 1
    s_xz = s_xz * 2 - 1
    s_xy = s_xy * 2 - 1

    y_yz = m_yz * s_yz * v_yz[:, 0:1, ...]
    z_yz = m_yz * s_yz * v_yz[:, 1:2, ...]
    x_xz = m_xz * s_xz * v_xz[:, 0:1, ...]
    z_xz = m_xz * s_xz * v_xz[:, 1:2, ...]
    x_xy = m_xy * s_xy * v_xy[:, 0:1, ...]
    y_xy = m_xy * s_xy * v_xy[:, 1:2, ...]

    x = (x_xz + x_xy) / 2
    y = (y_yz + y_xy) / 2
    z = (z_yz + z_xz) / 2

    v = torch.cat([x, y, z], dim=1)
    v = v / torch.norm(v, dim=1, keepdim=True)

    return v


def vector_2_piecewise_aligned(v: torch.Tensor):
    """
    Piecewise-Aligned representation
    v has shape (batch, 3, ...)
    returns a tensor of shape (batch, 9, ...)
    """
    v_x = vector_align(v, dim=0) * (v[:, 0:1, ...]**2)
    v_y = vector_align(v, dim=1) * (v[:, 1:2, ...]**2)
    v_z = vector_align(v, dim=2) * (v[:, 2:3, ...]**2)
    return torch.cat([v_x, v_y, v_z], dim=1)


@nan_to_zero
def piecewise_aligned_2_vector(r: torch.Tensor):
    """
    Piecewise-Aligned representation
    r has shape (batch, 9, ...)
    returns a tensor of shape (batch, 3, ...)
    """
    m_x = torch.norm(r[:, 0:3, ...], dim=1, keepdim=True)
    m_y = torch.norm(r[:, 3:6, ...], dim=1, keepdim=True)
    m_z = torch.norm(r[:, 6:9, ...], dim=1, keepdim=True)

    biggest_x = (m_x >= m_y) & (m_x >= m_z)
    biggest_y = (m_y > m_x) & (m_y >= m_z)
    biggest_z = (m_z > m_x) & (m_z > m_y)
    align_x = _sign(r[:, (0, 0, 0, 3, 3, 3, 6, 6, 6), ...])
    align_y = _sign(r[:, (1, 1, 1, 4, 4, 4, 7, 7, 7), ...])
    align_z = _sign(r[:, (2, 2, 2, 5, 5, 5, 8, 8, 8), ...])
    align = biggest_x * align_x + biggest_y * align_y + biggest_z*align_z

    r_aligned = r * align
    v = r_aligned[:, 0:3, ...] + r_aligned[:, 3:6, ...] + r_aligned[:, 6:9, ...]
    v = v / torch.norm(v, dim=1, keepdim=True)
    return v


@nan_to_zero
def vector_2_classification_dip_strike(v: torch.Tensor):
    """
    Classification Dip-Strike representation
    v has shape (batch, 3, ...)
    returns a tensor of shape (batch, 10, ...)
    """
    v_normalised = v / torch.norm(v, dim=1, keepdim=True)
    v_aligned = vector_align(v_normalised, 2)

    dip = torch.acos(v_aligned[:, 2:3, ...])
    dip_normalised = dip * 4 / pi
    strike = torch.atan2(-v_aligned[:, 1:2, ...], v_aligned[:, 0:1, ...])
    strike_normalised = torch.remainder(strike, 2*pi) * 3 / pi
    strike_i = torch.floor(strike_normalised).to(torch.long)

    w_dip = torch.remainder(dip_normalised, 1)
    w_strike = torch.remainder(strike_normalised, 1)

    w_lower_lower = (1 - w_dip) * (1 - w_strike)
    w_lower_upper = (1 - w_dip) * w_strike
    w_upper_lower = w_dip * (1 - w_strike)
    w_upper_upper = w_dip * w_strike

    representation = torch.zeros([v.shape[0], 10] + list(v.shape[2:]), dtype=v.dtype, device=v.device)

    # if dip_normalised > 1:
    dip_near_equator = dip_normalised > 1
    dip_near_pole = dip_normalised <= 1

    i_lower_lower = dip_near_equator * (strike_i + 3) + \
                    dip_near_pole * torch.full_like(strike_i, 9)
    i_lower_upper = dip_near_equator * (torch.remainder(strike_i + 1, 6) + 3) + \
                    dip_near_pole * torch.full_like(strike_i, 9)
    i_upper_lower = dip_near_equator * (torch.remainder(strike_i, 3)) + \
                    dip_near_pole * (strike_i + 3)
    i_upper_upper = dip_near_equator * (torch.remainder(strike_i + 1, 3)) + \
                    dip_near_pole * (torch.remainder(strike_i + 1, 6) + 3)

    representation.scatter_(1, i_lower_lower, w_lower_lower)
    representation.scatter_(1, i_lower_upper, w_lower_upper)
    representation.scatter_(1, i_upper_lower, w_upper_lower)
    representation.scatter_(1, i_upper_upper, w_upper_upper)

    representation[:, 9:10, ...] = (1 - w_dip) * dip_near_pole

    return representation


@nan_to_zero
def classification_dip_strike_2_vector(r: torch.Tensor):
    """
    Classification Dip-Strike representation
    r has shape (batch, 10, ...)
    returns a tensor of shape (batch, 3, ...)
    """
    r_halfj = torch.clone(r)
    r_halfj[:, 9, ...] /= 2  # will be summed two times later
    faces_vertices = torch.tensor(
        [[0, 1, 3, 4], [1, 2, 4, 5], [2, 0, 5, 6], [0, 1, 6, 7], [1, 2, 7, 8], [2, 0, 8, 3],
         [3, 4, 9, 9], [4, 5, 9, 9], [5, 6, 9, 9], [6, 7, 9, 9], [7, 8, 9, 9], [8, 3, 9, 9]],
        dtype=torch.long, device=r.device)
    faces_dips_lower = torch.tensor([pi/4] * 6 + [0] * 6, dtype=r.dtype, device=r.device)
    faces_strikes_lower = torch.tensor([0, pi/3, 2*pi/3, 3*pi/3, 4*pi/3, 5*pi/3] * 2, dtype=r.dtype, device=r.device)

    faces_probability = torch.zeros([r.shape[0], 12] + list(r.shape[2:]), dtype=r.dtype, device=r.device)
    for i_face, face_vertices in enumerate(faces_vertices):
        faces_probability[:, i_face, ...] = torch.sum(r_halfj[:, tuple(face_vertices), ...], dim=1)
    face_i = torch.argmax(faces_probability, dim=1, keepdim=True)

    faces_vertices_reshaped = faces_vertices.view([1, 12, 4] + [1]*(len(r.shape)-2))
    faces_dips_lower_reshaped = faces_dips_lower.view([1, 12, 1] + [1]*(len(r.shape)-2))
    faces_strikes_lower_reshaped = faces_strikes_lower.view([1, 12, 1] + [1]*(len(r.shape)-2))
    vertices = torch.take_along_dim(faces_vertices_reshaped, face_i.unsqueeze(2), dim=1)  # shape (batch, 1, 4, ...)
    dips_lower = torch.take_along_dim(faces_dips_lower_reshaped, face_i.unsqueeze(2), dim=1)[:, 0, :, ...]
    strikes_lower = torch.take_along_dim(faces_strikes_lower_reshaped, face_i.unsqueeze(2), dim=1)[:, 0, :, ...]

    weights = torch.take_along_dim(r_halfj.unsqueeze(2), vertices, dim=1)[:, 0, :, ...]
    weights_normalised = weights / torch.sum(weights, dim=1, keepdim=True)

    dip = dips_lower + (pi/4) * (weights_normalised[:, 0:1, ...] + weights_normalised[:, 1:2, ...])
    strike_equator = strikes_lower + (pi/3) * (weights_normalised[:, 1:2, ...] + weights_normalised[:, 3:4, ...])
    strike_pole = strikes_lower + (pi/3) * weights[:, 1:2, ...] / (weights[:, 0:1, ...] + weights[:, 1:2, ...])
    strike = strike_equator * (face_i < 6) + strike_pole * (face_i >= 6)

    z = torch.cos(dip)
    sin_dip = torch.sin(dip)
    x = torch.cos(strike) * sin_dip
    y = -torch.sin(strike) * sin_dip
    v = torch.cat([x, y, z], dim=1)
    v = v / torch.norm(v, dim=1, keepdim=True)

    return v


@lru_cache
def _get_icosahedron_faces_vertices(device='cpu'):
    """ helper function for classification icosahedron """
    faces_vertices = torch.LongTensor([
        [1, 0, 2],
        [0, 1, 3],
        [0, 2, 4],
        [4, 5, 0],
        [0, 3, 5],
        [1, 3, 10],
        [10, 11, 1],
        [1, 2, 11],
        [2, 9, 11],
        [2, 9, 4]
    ])
    return torch.cat([faces_vertices, (faces_vertices + 6) % 12]).to(device=device)


def _get_icosahedron_faces_vertices_coordinates(dtype=torch.float32, device='cpu'):
    """ helper function for classification icosahedron """
    golden = (1 + sqrt(5)) / 2
    vertices_coordinates = torch.tensor([
        [0, 1, golden],
        [0, -1, golden],
        [golden, 0, 1],
        [-golden, 0, 1],
        [1, golden, 0],
        [-1, golden, 0]
    ], dtype=dtype)
    vertices_coordinates = torch.cat([vertices_coordinates, -vertices_coordinates], dim=0)

    faces_vertices = _get_icosahedron_faces_vertices()
    return torch.take_along_dim(vertices_coordinates[None, :, :], faces_vertices[:, :, None], dim=1).to(device=device)


@lru_cache
def _get_icosahedron_faces_centres(device='cpu'):
    """ helper function for classification icosahedron """
    faces_vertices_coordinates = _get_icosahedron_faces_vertices_coordinates()

    faces_centres = torch.sum(faces_vertices_coordinates, dim=1)
    faces_centres = faces_centres / torch.norm(faces_centres, dim=1, keepdim=True)
    return faces_centres.to(device=device)


@lru_cache
def _get_icosahedron_transformation_matrices(dtype=torch.float32, device='cpu'):
    """ helper function for classification icosahedron """
    def get_translation_matrix(x, y, z):
        return torch.tensor([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [x, y, z, 1]
        ], dtype=dtype, device=device)

    def get_rotation_matrix_z(angle):
        return torch.tensor([
            [math.cos(angle), math.sin(angle), 0, 0],
            [-math.sin(angle), math.cos(angle), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=dtype, device=device)

    def get_rotation_matrix_y(angle):
        return torch.tensor([
            [math.cos(angle), 0, -math.sin(angle), 0],
            [0, 1, 0, 0],
            [math.sin(angle), 0, math.cos(angle), 0],
            [0, 0, 0, 1]
        ], dtype=dtype, device=device)

    def get_rotation_matrix_x(angle):
        return torch.tensor([
            [1, 0, 0, 0],
            [0, math.cos(angle), math.sin(angle), 0],
            [0, -math.sin(angle), math.cos(angle), 0],
            [0, 0, 0, 1]
        ], dtype=dtype, device=device)

    faces_vertices_coordinates = _get_icosahedron_faces_vertices_coordinates(dtype=dtype)
    faces_vertices_coordinates_expanded = torch.cat([faces_vertices_coordinates[:10, :, :], torch.ones((10, 3, 1), dtype=dtype)], dim=2)

    face_transformation_matrices = []
    face_transformation_inverse_matrices = []
    for face_i in range(len(faces_vertices_coordinates_expanded)):
        v0 = faces_vertices_coordinates_expanded[face_i]
        t0 = get_translation_matrix(-v0[0][0], -v0[0][1], -v0[0][2])
        t0_inv = get_translation_matrix(v0[0][0], v0[0][1], v0[0][2])
        v1 = v0 @ t0
        t1 = get_rotation_matrix_y(math.atan2(v1[1][2], v1[1][0]))
        t1_inv = get_rotation_matrix_y(-math.atan2(v1[1][2], v1[1][0]))
        v2 = v1 @ t1
        t2 = get_rotation_matrix_z(-math.atan2(v2[1][1], v2[1][0]))
        t2_inv = get_rotation_matrix_z(math.atan2(v2[1][1], v2[1][0]))
        v3 = v2 @ t2
        t3 = get_rotation_matrix_x(-math.atan2(v3[2][2], v3[2][1]))
        t3_inv = get_rotation_matrix_x(math.atan2(v3[2][2], v3[2][1]))
        # v4 = v3 @ t3
        t = t0 @ t1 @ t2 @ t3
        t_inv = t3_inv @ t2_inv @ t1_inv @ t0_inv
        face_transformation_matrices.append(t)
        face_transformation_inverse_matrices.append(t_inv)

    face_transformation_matrices = torch.stack(face_transformation_matrices)
    face_transformation_inverse_matrices = torch.stack(face_transformation_inverse_matrices)

    return face_transformation_matrices.to(dtype=dtype, device=device), face_transformation_inverse_matrices.to(dtype=dtype, device=device)


def _apply_transformations(vectors, transformations):
    """ Apply transformations of shape (batch, 4, 4, ...) to vectors of shape (batch, 3, ...) """
    shape_no_vector = list(vectors.shape)
    del shape_no_vector[1]
    num_broadcast_flat = math.prod(shape_no_vector)
    vectors_flat = vectors.movedim(1, -1).reshape(-1, 1, 3)
    vectors_flat = torch.cat([vectors_flat, torch.ones([num_broadcast_flat, 1, 1], dtype=vectors.dtype, device=vectors.device)], dim=-1)
    transformations_flat = transformations.movedim(1, -1).movedim(1, -1).reshape(-1, 4, 4)
    result_flat = torch.bmm(vectors_flat, transformations_flat)
    result_flat = result_flat[:, 0, 0:3]
    result = result_flat.reshape(shape_no_vector + [3]).movedim(-1, 1)
    return result


@nan_to_zero
def _cosine_similarity(v0, v1, dim=1, keepdim=True):
    """ helper function for classification icosahedron """
    dot_product = torch.sum(v0 * v1, dim=dim, keepdim=keepdim)
    m0 = torch.norm(v0, dim=dim, keepdim=keepdim)
    m1 = torch.norm(v1, dim=dim, keepdim=keepdim)
    similarity = dot_product / (m0 * m1)
    similarity = torch.clamp(similarity, -1, 1)
    return similarity


@nan_to_zero
def vector_2_classification_icosahedron(v: torch.Tensor):
    """"
    Classification Icosahedron representation
    v has shape (batch, 3, ...)
    returns a tensor of shape (batch, 6, ...)
    """
    # Step 1: Find which face vectors belong to
    n_broadcast_dims = len(v.shape) - 2
    faces_centres = _get_icosahedron_faces_centres(device=v.device)
    faces_centres_broadcast_shape = [1, 20, 3] + [1] * n_broadcast_dims
    faces_centres_broadcast = faces_centres.view(faces_centres_broadcast_shape)
    faces_centres_similarity = _cosine_similarity(v[:, None, :, ...], faces_centres_broadcast, dim=2, keepdim=False)
    face_i = torch.argmax(faces_centres_similarity, dim=1, keepdim=True)

    # Step 2: Negate vectors that belong to faces 10:20
    face_i_ge_10 = face_i >= 10
    v = v * -(face_i_ge_10 * 2 - 1)
    face_i = face_i - 10 * face_i_ge_10

    # Step 3: Transform vector onto centered face
    n_broadcast_dims = len(v.shape) - 2
    face_transformation_matrices, _, = _get_icosahedron_transformation_matrices(dtype=v.dtype, device=v.device)
    face_transformation_matrices_broadcast = face_transformation_matrices.view([1, 10, 4, 4] + [1] * n_broadcast_dims)
    transformation_matrices = torch.take_along_dim(face_transformation_matrices_broadcast, face_i[:, :, None, None, ...], dim=1)[:, 0, :, :, ...]
    transformed_vectors = _apply_transformations(v, transformation_matrices)
    transformed_origins = _apply_transformations(torch.zeros_like(v), transformation_matrices)
    # find transformed_vectors on face (with z == 0)
    coefficient = transformed_origins[:, 2:3, ...] / (transformed_origins[:, 2:3, ...] - transformed_vectors[:, 2:3, ...])
    vectors_on_face = transformed_origins + coefficient * (transformed_vectors - transformed_origins)

    # Step 4: Find representation coefficients for points on face
    repr_a = 1 - vectors_on_face[:, 0:1, ...] / 2 - vectors_on_face[:, 1:2, ...] / (2 * math.sqrt(3))
    repr_b = vectors_on_face[:, 0:1, ...] / 2 - vectors_on_face[:, 1:2, ...] / (2 * math.sqrt(3))
    repr_c = vectors_on_face[:, 1:2, ...] / (math.sqrt(3))
    representation_coefficients = torch.cat([repr_a, repr_b, repr_c], dim=1)

    # Step 5: Assign representation coefficients to representation result
    n_broadcast_dims = len(representation_coefficients.shape) - 2
    faces_vertices = _get_icosahedron_faces_vertices(device=representation_coefficients.device)[0:10]
    faces_vertices = torch.remainder(faces_vertices, 6)  # we only want one vertex from each +/- pair
    faces_vertices_broadcast = faces_vertices.view([1, 10, 3] + [1] * n_broadcast_dims)

    vertices = torch.take_along_dim(faces_vertices_broadcast, face_i[:, :, None, ...], dim=1)[:, 0, :, ...]

    result_shape = list(face_i.shape)
    result_shape[1] = 6
    representation = torch.zeros(result_shape, dtype=representation_coefficients.dtype, device=face_i.device)
    representation.scatter_(1, vertices, representation_coefficients)

    return representation


@nan_to_zero
def classification_icosahedron_2_vector(r: torch.Tensor):
    """"
    Classification Icosahedron representation
    r has shape (batch, 6, ...)
    returns a tensor of shape (batch, 3, ...)
    """
    # Step 1: Find most likely face
    n_broadcast_dims = len(r.shape) - 2
    faces_vertices = _get_icosahedron_faces_vertices(device=r.device)[0:10, :]
    faces_vertices = torch.remainder(faces_vertices, 6)  # we only want one vertex from each +/- pair
    faces_vertices_broadcast = faces_vertices.view([1, 10, 3] + [1] * n_broadcast_dims)
    faces_vertices_coeff = torch.take_along_dim(r[:, :, None, ...], faces_vertices_broadcast, dim=1)
    faces_coeff = torch.sum(faces_vertices_coeff, dim=2)
    face_i = torch.argmax(faces_coeff, dim=1, keepdim=True)
    vertices_coeff = torch.take_along_dim(faces_vertices_coeff, face_i[:, :, None], dim=1)[:, 0, :]

    # Step 2: Find vectors on centered face
    vertices_coeff = vertices_coeff / torch.sum(vertices_coeff, dim=1, keepdim=True)
    x = 1 - vertices_coeff[:, 0:1, ...] + vertices_coeff[:, 1:2, ...]
    y = math.sqrt(3) * vertices_coeff[:, 2:3, ...]
    vectors_on_face = torch.cat([x, y, torch.zeros_like(x)], dim=1)

    # Step 3: Transform vector on centered face to vector on icosahedron
    n_broadcast_dims = len(vectors_on_face.shape) - 2
    _, face_transformation_matrices_inverse, = _get_icosahedron_transformation_matrices(dtype=r.dtype, device=vectors_on_face.device)
    face_transformation_matrices_inverse_broadcast = face_transformation_matrices_inverse.view([1, 10, 4, 4] + [1] * n_broadcast_dims)
    transformation_matrices = torch.take_along_dim(face_transformation_matrices_inverse_broadcast, face_i[:, :, None, None, ...], dim=1)[:, 0, :, :, ...]
    transformed_vectors = _apply_transformations(vectors_on_face, transformation_matrices)

    # Step 4: Normalise vector
    vectors = transformed_vectors / torch.norm(transformed_vectors, dim=1, keepdim=True)
    return vectors


def vector_2_saxena(v: torch.Tensor):
    """"
    Saxena's representation
    v has shape (batch, 3, ...)
    returns a tensor of shape (batch, 6, ...)
    """
    representation = v.unsqueeze(2) * v.unsqueeze(1)
    representation = representation.view(v.shape[0], 9, *v.shape[2:])
    representation = representation[:, (0, 1, 2, 4, 5, 8), ...]
    return representation


def saxena_2_vector(r: torch.Tensor):
    """"
    Saxena's representation
    r has shape (batch, 6, ...)
    returns a tensor of shape (batch, 3, ...)
    """
    A = r[:, (0, 1, 2, 1, 3, 4, 2, 4, 5), ...]
    A = torch.moveaxis(A, 1, -1)
    A = A.view(r.shape[0], *r.shape[2:], 3, 3)
    vector = torch.pca_lowrank(A)[2][..., 0]
    vector = torch.moveaxis(vector, -1, 1)

    return vector
