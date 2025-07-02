import numpy as np
import model_constants

def get_service_time(job_type, operation_idx, mean_service_times):
    """Generate service time using gamma distribution with shape=2."""
    # station where the machine is located
    station = model_constants.ROUTINGS[job_type][operation_idx]
    if ((model_constants.MACHINE_EFFICIENCY != 1) and (station in (model_constants.EFFICENCY_TARGET_MACHINES))):
        efficiency = model_constants.MACHINE_EFFICIENCY
    else:
        efficiency = 1

    # print("Machine station:", station)
    # print("Machine efficiency: ", efficiency)
    mean = mean_service_times[job_type][operation_idx]/efficiency
    shape = 2
    scale = mean / shape
    return np.random.gamma(shape, scale)

class Machine:
    def __init__(self, machine_type=None, mean_service_times=None):
        self.state = 'idle'
        self.current_job = None
        self.service_time = 0
        self.machine_type = machine_type
        self.mean_service_times = mean_service_times
        
        # Statistics
        self.total_busy_time = 0
        self.total_blocked_time = 0
        self.total_idle_time = 0
        self.jobs_processed = 0
        self.last_state_change_time = 0

    def assign_job(self, job, operation_idx, current_time=None):
        """Assign a job to this machine and start processing."""
        if self.state == 'idle':
            if current_time is not None:
                self.update_state_time(current_time, 'busy')
            else:
                self.state = 'busy'
            
            self.current_job = job
            self.service_time = get_service_time(job.job_type, operation_idx, self.mean_service_times)
            return self.service_time
        return None

    def finish_processing(self, current_time=None):
        """Mark job as finished, machine becomes blocked waiting for pickup."""
        if self.state == 'busy':
            if current_time is not None:
                self.update_state_time(current_time, 'blocked')
            else:
                self.state = 'blocked'
            self.jobs_processed += 1

    def remove_job(self, current_time=None):
        """Remove job from machine (pickup by forklift), machine becomes idle."""
        if self.state == 'blocked':
            if current_time is not None:
                self.update_state_time(current_time, 'idle')
            else:
                self.state = 'idle'
            self.current_job = None
            self.service_time = 0

    def update_state_time(self, current_time, new_state=None):
        """Update time tracking when state changes."""
        if hasattr(self, 'last_state_change_time'):
            time_diff = current_time - self.last_state_change_time
            
            if self.state == 'busy':
                self.total_busy_time += time_diff
            elif self.state == 'blocked':
                self.total_blocked_time += time_diff
            elif self.state == 'idle':
                self.total_idle_time += time_diff
        
        self.last_state_change_time = current_time
        if new_state:
            self.state = new_state
    
    def get_utilization_stats(self, total_time):
        """Calculate machine utilization statistics."""
        if total_time <= 0:
            return {'busy': 0, 'blocked': 0, 'idle': 0}
        
        return {
            'busy': self.total_busy_time / total_time,
            'blocked': self.total_blocked_time / total_time,
            'idle': self.total_idle_time / total_time
        }

