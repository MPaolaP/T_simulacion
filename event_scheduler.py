import heapq

class Event:
    """Represents a simulation event."""
    
    def __init__(self, time, event_type, **kwargs):
        self.time = time
        self.event_type = event_type
        self.kwargs = kwargs
    
    def __lt__(self, other):
        return self.time < other.time
    
    def __repr__(self):
        return f"Event({self.time:.3f}, {self.event_type}, {self.kwargs})"

class EventScheduler:
    """Manages the event list for the simulation."""
    
    def __init__(self):
        self.event_list = []
        self.current_time = 0
    
    def schedule(self, time, event_type, **kwargs):
        """Schedule a new event."""
        event = Event(time, event_type, **kwargs)
        heapq.heappush(self.event_list, event)
        return event
    
    def next_event(self):
        """Get the next event from the list."""
        if self.event_list:
            event = heapq.heappop(self.event_list)
            self.current_time = event.time
            return event
        return None
    
    def has_events(self):
        """Check if there are more events."""
        return len(self.event_list) > 0
    
    def peek_next_time(self):
        """Get the time of the next event without removing it."""
        if self.event_list:
            return self.event_list[0].time
        return float('inf')
