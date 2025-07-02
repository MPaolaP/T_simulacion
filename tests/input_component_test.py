import unittest
import numpy as np
from input_component import JobInputGenerator

class TestJobInputGenerator(unittest.TestCase):
    def setUp(self):
        self.mean = 1/15
        self.job_types = [1, 2, 3]
        self.probs = [0.3, 0.5, 0.2]
        self.generator = JobInputGenerator(self.mean, self.job_types, self.probs)
        np.random.seed(42)  # For reproducibility

    def test_job_type_values(self):
        for _ in range(100):
            self.assertIn(self.generator.next_job_type(), self.job_types)

if __name__ == '__main__':
    unittest.main()