import unittest
from unittest.mock import patch

from machine import Machine

class DummyJob:
    def __init__(self, job_type):
        self.job_type = job_type

class TestMachine(unittest.TestCase):
    def setUp(self):
        self.mean_service_times = {1: [0.25, 0.15, 0.10, 0.30], 2: [0.15, 0.20, 0.30], 3: [0.15, 0.10, 0.35, 0.20, 0.20]}
        self.machine = Machine(machine_type='A', mean_service_times=self.mean_service_times)
        self.job = DummyJob(1)  # Use numeric job type for mean_service_times

    def test_initial_state(self):
        # GIVEN a new machine
        # THEN it should be idle, with no job and zero service time
        self.assertEqual(self.machine.state, 'idle')
        self.assertIsNone(self.machine.current_job)
        self.assertEqual(self.machine.service_time, 0)

    @patch('machine.get_service_time', return_value=2.5)
    def test_assign_job(self, mock_service_time):
        # GIVEN an idle machine
        # WHEN a job is assigned
        # THEN the machine should be busy, store the job, and set the service time
        self.machine.assign_job(self.job, 0)
        self.assertEqual(self.machine.state, 'busy')
        self.assertEqual(self.machine.current_job, self.job)
        self.assertEqual(self.machine.service_time, 2.5)

    @patch('machine.get_service_time', return_value=1.0)
    def test_finish_processing(self, mock_service_time):
        # GIVEN a busy machine
        # WHEN finish_processing is called
        # THEN the machine should become blocked
        self.machine.assign_job(self.job, 0)
        self.machine.finish_processing()
        self.assertEqual(self.machine.state, 'blocked')

    @patch('machine.get_service_time', return_value=1.0)
    def test_remove_job(self, mock_service_time):
        # GIVEN a blocked machine
        # WHEN remove_job is called
        # THEN the machine should become idle, with no job and zero service time
        self.machine.assign_job(self.job, 0)
        self.machine.finish_processing()
        self.machine.remove_job()
        self.assertEqual(self.machine.state, 'idle')
        self.assertIsNone(self.machine.current_job)
        self.assertEqual(self.machine.service_time, 0)

if __name__ == '__main__':
    unittest.main()
