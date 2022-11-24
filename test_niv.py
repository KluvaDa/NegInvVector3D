import torch
import niv

if __name__ == '__main__':
    batch_size = 1000
    distribute_shape = (4, 5, 6)
    random_vector_64 = torch.rand((batch_size, 3) + distribute_shape, dtype=torch.float64) * 2 - 1
    random_vector_64 = random_vector_64 / torch.norm(random_vector_64, dim=1, keepdim=True)
    random_vector_32 = random_vector_64.to(dtype=torch.float32)

    for vector, precision in ((random_vector_64, 64), (random_vector_32, 32)):
        for vector_2_repr, repr_2_vector in (
                (niv.vector_2_dip90_strike360, niv.dip90_strike360_2_vector),
                (niv.vector_2_dip180_strike180, niv.dip180_strike180_2_vector),
                (niv.vector_2_projection_doubleangle, niv.projection_doubleangle_2_vector),
                (niv.vector_2_piecewise_align, niv.piecewise_align_2_vector),
                (niv.vector_2_classification_dip_strike, niv.classification_dip_strike_2_vector),
                (niv.vector_2_classification_icosahedron, niv.classification_icosahedron_2_vector),
                (niv.vector_2_saxena, niv.saxena_2_vector)
        ):
            representation = vector_2_repr(vector)
            vector_recreation = repr_2_vector(representation)
            vector_recreation_norm = torch.norm(vector_recreation, dim=1, keepdim=True)
            if precision == 64:
                atol = 1e-11
            else:
                atol = 2e-4
            # magnitude
            assert torch.all(torch.isclose(vector_recreation_norm,
                                           torch.full((1,)*5, dtype=vector.dtype, fill_value=1.0),
                                           atol=atol,
                                           rtol=0))
            # recreation
            difference = torch.max(torch.minimum(torch.norm(vector-vector_recreation, dim=1), torch.norm(vector+vector_recreation, dim=1)))
            print(torch.max(difference))
            assert torch.all(torch.isclose(difference,
                                           torch.full((1,) * 5, dtype=vector.dtype, fill_value=0.0),
                                           atol=atol,
                                           rtol=0))
