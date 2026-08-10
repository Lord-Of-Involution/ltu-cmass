import unittest

import numpy as np

from cmass.diagnostics.calculations import (
    K_MAX_PK,
    _fixed_pk_kedges,
    fixed_lightcone_pk_edges,
)


class FixedPkGridTest(unittest.TestCase):
    def test_fixed_pk_grid_has_expected_physical_edges(self):
        edges = fixed_lightcone_pk_edges()

        self.assertEqual(edges.shape, (80,))
        self.assertTrue(np.allclose(edges, np.arange(0.01, 0.801, 0.01)))
        self.assertTrue(np.isclose(edges[-1], K_MAX_PK))
        self.assertTrue(np.allclose(_fixed_pk_kedges(0.81), edges))

    def test_fixed_pk_grid_rejects_insufficient_nyquist(self):
        with self.assertRaisesRegex(ValueError, 'exceeds.*Nyquist'):
            _fixed_pk_kedges(0.79)


if __name__ == '__main__':
    unittest.main()
