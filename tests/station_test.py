import unittest
from station import Station

class DummyJob:
    def __init__(self, job_type):
        self.job_type = job_type

class TestStation(unittest.TestCase):
    def setUp(self):
        # GIVEN a station with 2 machines and constant mean service times
        mean_service_times = {1: [1.0, 1.0, 1.0, 1.0], 2: [1.0, 1.0, 1.0], 3: [1.0, 1.0, 1.0, 1.0, 1.0]}
        self.station = Station(station_id=1, num_machines=2, mean_service_times=mean_service_times)

    def test_arrive_when_machine_free(self):
        # GIVEN a station with at least one idle machine
        job = DummyJob(1)
        # WHEN a job arrives
        self.station.arrive(job, operation_idx=0)
        # THEN one machine should be busy and the queue should be empty
        busy_count = sum(1 for m in self.station.machines if m.state == 'busy')
        self.assertEqual(busy_count, 1)
        self.assertEqual(len(self.station.queue), 0)

    def test_arrive_when_all_busy(self):
        # GIVEN a station where all machines are busy
        for m in self.station.machines:
            m.state = 'busy'
        job = DummyJob(2)
        # WHEN a job arrives
        self.station.arrive(job, operation_idx=0)
        # THEN the job should be queued
        self.assertEqual(len(self.station.queue), 1)

    def test_finish_processing_blocks_machine(self):
        # GIVEN a station with a busy machine
        job = DummyJob(1)
        self.station.arrive(job, operation_idx=0)
        # WHEN finish_processing is called on that machine
        self.station.finish_processing(0)
        # THEN the machine should be blocked
        self.assertEqual(self.station.machines[0].state, 'blocked')

    def test_remove_job_unblocks_and_processes_next(self):
        # GIVEN a station with one blocked machine and a queued job
        self.station.machines[0].state = 'blocked'
        self.station.machines[1].state = 'busy'
        self.station.queue.append((DummyJob(3), 0))
        # WHEN remove_job is called on the blocked machine
        self.station.remove_job(0)
        # THEN the machine should become busy and the queue should be empty
        self.assertEqual(self.station.machines[0].state, 'busy')
        self.assertEqual(len(self.station.queue), 0)

if __name__ == '__main__':
    unittest.main()