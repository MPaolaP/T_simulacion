import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_constants import DISTANCES, FORKLIFT_SPEED
from forklift import Forklift

class TestForklift(unittest.TestCase):
    def setUp(self):
        # GIVEN two forklifts at different stations
        self.f1 = Forklift(forklift_id=1, current_station=1)
        self.f2 = Forklift(forklift_id=2, current_station=3)
        self.forklifts = [self.f1, self.f2]
        self.current_time = 0

    def test_assign_transport_sets_state_and_time(self):
        # GIVEN an idle forklift at station 1
        # WHEN assigned to move from station 1 to 2
        finish_time = self.f1.assign_transport(from_station=1, to_station=2, current_time=self.current_time)
        
        # THEN the forklift should not be idle and time should be calculated correctly
        # Time = (distance to pickup + distance to destination) / speed / 3600 (convert to hours)
        expected_time = (DISTANCES[0][0] + DISTANCES[0][1]) / FORKLIFT_SPEED / 3600
        
        self.assertFalse(self.f1.is_idle)
        self.assertAlmostEqual(self.f1.time_until_idle, expected_time, places=6)
        self.assertEqual(self.f1.current_station, 2)
        self.assertAlmostEqual(finish_time, expected_time, places=6)

    def test_become_idle_resets_state(self):
        # GIVEN a forklift in transit
        self.f1.assign_transport(1, 2, self.current_time)
        initial_time = self.f1.time_until_idle
        
        # WHEN forklift becomes idle
        self.f1.become_idle(self.current_time + initial_time)
        
        # THEN the forklift should be idle with zero time
        self.assertTrue(self.f1.is_idle)
        self.assertEqual(self.f1.time_until_idle, 0)

    def test_distance_calculation(self):
        # GIVEN a forklift at station 1
        # WHEN calculating distance to station 3
        distance = self.f1.distance_to_station(3)
        
        # THEN distance should match the distance matrix
        expected_distance = DISTANCES[0][2]  # station 1 to station 3
        self.assertEqual(distance, expected_distance)

    def test_utilization_tracking(self):
        # GIVEN a forklift
        # WHEN it performs transport
        self.f1.assign_transport(1, 2, 0)
        finish_time = self.f1.time_until_idle
        self.f1.become_idle(finish_time)
        
        # THEN utilization should be calculated correctly
        utilization = self.f1.get_utilization(finish_time)
        self.assertAlmostEqual(utilization, 1.0, places=6)  # 100% utilized during transport

if __name__ == "__main__":
    unittest.main()

if __name__ == "__main__":
    unittest.main()
