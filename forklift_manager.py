from collections import deque

class ForkliftRequest:
    """Represents a request for forklift service."""
    
    def __init__(self, job, from_station, to_station, request_time, operation_idx=None):
        self.job = job
        self.from_station = from_station
        self.to_station = to_station
        self.request_time = request_time
        self.operation_idx = operation_idx
    
    def __repr__(self):
        return f"ForkliftRequest(job={self.job.job_id}, from={self.from_station}, to={self.to_station})"

class ForkliftManager:
    """Manages forklift allocation and request queuing."""
    
    def __init__(self, forklifts):
        self.forklifts = forklifts
        self.request_queue = deque()
        self.total_requests = 0
        self.total_wait_time = 0
    
    def request_forklift(self, job, from_station, to_station, current_time, operation_idx=None):
        """Request a forklift for transportation."""
        request = ForkliftRequest(job, from_station, to_station, current_time, operation_idx)
        self.request_queue.append(request)
        self.total_requests += 1
        return self.try_assign_forklift(current_time)
    
    def try_assign_forklift(self, current_time):
        """Try to assign an idle forklift to the next request."""
        if not self.request_queue:
            return None
        
        # Get idle forklifts
        idle_forklifts = [f for f in self.forklifts if f.is_idle]
        if not idle_forklifts:
            return None
        
        # Get the next request (FIFO, but we could implement shortest distance first)
        request = self.request_queue.popleft()
        
        # Find closest forklift to pickup location
        closest_forklift = min(idle_forklifts, 
                             key=lambda f: f.distance_to_station(request.from_station))
        
        # Calculate wait time
        wait_time = current_time - request.request_time
        self.total_wait_time += wait_time
        
        # Assign the forklift
        finish_time = closest_forklift.assign_transport(
            request.from_station, request.to_station, current_time)
        
        return {
            'forklift': closest_forklift,
            'request': request,
            'finish_time': finish_time,
            'wait_time': wait_time
        }
    
    def forklift_became_idle(self, forklift, current_time):
        """Called when a forklift becomes idle to try processing queued requests."""
        forklift.is_idle = True
        return self.try_assign_forklift(current_time)
    
    def get_queue_stats(self):
        """Get statistics about the forklift request queue."""
        return {
            'current_queue_length': len(self.request_queue),
            'total_requests': self.total_requests,
            'average_wait_time': self.total_wait_time / self.total_requests if self.total_requests > 0 else 0
        }
