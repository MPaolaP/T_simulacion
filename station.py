from machine import Machine
import numpy as np

class Station:
    def __init__(self, station_id, num_machines, mean_service_times):
        self.station_id = station_id
        self.num_machines = num_machines
        self.queue = []
        self.machines = [Machine(machine_type=station_id, mean_service_times=mean_service_times) 
                        for _ in range(num_machines)]
        
        # Statistics
        self.stats = {
            'total_arrivals': 0,
            'total_departures': 0,
            'total_queue_time': 0,
            'max_queue_length': 0,
            'queue_length_samples': [],
            'jobs_processed': 0
        }
        self.last_update_time = 0

    def arrive(self, job, operation_idx, current_time):
        """A job arrives at this station."""
        self.stats['total_arrivals'] += 1
        job.log(f'arrived at station {self.station_id}', current_time, 
               f'operation {operation_idx}, queue length: {len(self.queue)}')
        
        # Try to assign to an idle machine
        for i, machine in enumerate(self.machines):
            if machine.state == 'idle':
                service_time = machine.assign_job(job, operation_idx, current_time)
                job.log(f'started processing at machine {i}', current_time)
                return service_time  # Return processing time
        
        # No idle machine, add to queue
        job.queue_entry_time = current_time
        self.queue.append((job, operation_idx))
        self.update_queue_stats()
        return None  # Job queued, no processing time yet

    def finish_processing(self, machine_idx, current_time):
        """A machine finishes processing a job."""
        machine = self.machines[machine_idx]
        job = machine.current_job
        
        if job:
            job.log(f'finished processing at station {self.station_id}', current_time,
                   f'machine {machine_idx}')
            self.stats['jobs_processed'] += 1
            
        # Mark machine as blocked (waiting for forklift pickup)
        machine.finish_processing(current_time)
        
        return job

    def remove_job(self, machine_idx, current_time):
        """Remove job from blocked machine (forklift pickup)."""
        machine = self.machines[machine_idx]
        job = machine.current_job
        
        if job:
            job.log(f'picked up from station {self.station_id}', current_time,
                   f'machine {machine_idx}')
            self.stats['total_departures'] += 1
        
        # Unblock machine
        machine.remove_job(current_time)
        
        # Process next job in queue if any
        processing_time = None
        if self.queue and machine.state == 'idle':
            next_job, operation_idx = self.queue.pop(0)
            
            # Calculate queue time
            queue_time = current_time - next_job.queue_entry_time
            self.stats['total_queue_time'] += queue_time
            next_job.time_in_queue += queue_time
            
            machine.assign_job(next_job, operation_idx, current_time)
            next_job.log(f'started processing at machine {machine_idx}', current_time,
                        f'waited in queue: {queue_time:.3f}h')
            processing_time = machine.service_time
            
            self.update_queue_stats()
        
        return job, processing_time

    def get_blocked_jobs(self):
        """Get all jobs currently blocked (finished processing, waiting for pickup)."""
        blocked_jobs = []
        for i, machine in enumerate(self.machines):
            if machine.state == 'blocked' and machine.current_job:
                blocked_jobs.append((machine.current_job, i))
        return blocked_jobs

    def update_queue_stats(self):
        """Update queue length statistics."""
        current_length = len(self.queue)
        self.stats['queue_length_samples'].append(current_length)
        if current_length > self.stats['max_queue_length']:
            self.stats['max_queue_length'] = current_length

    def get_machine_states(self):
        """Get current state of all machines."""
        return [m.state for m in self.machines]

    def get_utilization_stats(self, total_time):
        """Calculate utilization statistics for this station."""
        if self.num_machines == 0:  # I/O station
            return {
                'busy': 0, 'blocked': 0, 'idle': 0,
                'average_queue_length': 0,
                'max_queue_length': 0,
                'jobs_processed': 0
            }
        
        if total_time == 0 or not self.machines:
            busy_util = blocked_util = idle_util = 0
        else:
            # Calculate actual utilization from machine statistics
            total_busy = sum(m.total_busy_time for m in self.machines)
            total_blocked = sum(m.total_blocked_time for m in self.machines)
            total_idle = sum(m.total_idle_time for m in self.machines)
            total_machine_time = total_time * self.num_machines
            
            if total_machine_time > 0:
                busy_util = total_busy / total_machine_time
                blocked_util = total_blocked / total_machine_time
                idle_util = total_idle / total_machine_time
            else:
                busy_util = blocked_util = idle_util = 0
        
        avg_queue_length = (
            sum(self.stats['queue_length_samples']) / len(self.stats['queue_length_samples']) 
            if self.stats['queue_length_samples'] else 0
        )
        
        return {
            'busy': busy_util,
            'blocked': blocked_util,
            'idle': idle_util,
            'average_queue_length': avg_queue_length,
            'max_queue_length': self.stats['max_queue_length'],
            'jobs_processed': self.stats['jobs_processed']
        }
