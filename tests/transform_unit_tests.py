"""Do some basic tests."""

import unittest

class TransformTests(unittest.TestCase):

    def test_import_max_intensity_projection(self):
        # This throws an error if the function cannot be imported.
        from jicbioimage.transform import max_intensity_projection

    def test_import_min_intensity_projection(self):
        # This throws an error if the function cannot be imported.
        from jicbioimage.transform import min_intensity_projection

    def test_import_smooth_gaussian(self):
        # This throws an error if the function cannot be imported.
        from jicbioimage.transform import smooth_gaussian

    def test_import_remove_small_objects(self):
        # This throws an error if the function cannot be imported.
        from jicbioimage.transform import remove_small_objects

    def test_threshold_otsu(self):
        # This throws an error if the function cannot be imported.
        from jicbioimage.transform import threshold_otsu

if __name__ == '__main__':
    unittest.main()
