class Job:
    """Represents a job in the manufacturing system."""
    
    def __init__(self, job_id, job_type, routing):
        self.job_id = job_id
        self.job_type = job_type
        self.routing = routing
        self.current_op = 0
        self.arrival_time = None
        self.finish_time = None
        self.trace = []
        
        # Track time spent in different states
        self.time_in_queue = 0.0
        self.time_being_processed = 0.0
        self.time_being_transported = 0.0
        
        # For queue tracking
        self.queue_entry_time = None
        
    def get_current_station(self):
        """Get the station where the job should be processed for its current operation."""
        if self.current_op < len(self.routing):
            return self.routing[self.current_op]
        return 6  # I/O station (exit)
    
    def get_next_station(self):
        """Get the next station in the job's routing."""
        if self.current_op + 1 < len(self.routing):
            return self.routing[self.current_op + 1]
        return 6  # I/O station (exit)
    
    def advance_operation(self):
        """Move to the next operation in the routing."""
        self.current_op += 1
    
    def is_complete(self):
        """Check if the job has completed all operations."""
        return self.current_op >= len(self.routing)
    
    def log(self, event, time, extra=None):
        """Log an event for this job."""
        msg = f"[t={time:.3f}] Job {self.job_id} (type {self.job_type}) - {event}"
        if extra:
            msg += f" | {extra}"
        self.trace.append(msg)
        # print(msg)  # Comment out for less verbose output
