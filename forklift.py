from model_constants import DISTANCES, FORKLIFT_SPEED

class Forklift:
    def __init__(self, forklift_id, current_station=6):
        self.forklift_id = forklift_id
        self.current_station = current_station  # Station numbers are 1-based
        self.is_idle = True
        self.time_until_idle = 0
        self.trace = []
        
        # Statistics
        self.total_transport_time = 0
        self.total_idle_time = 0
        self.total_empty_time = 0
        self.total_loaded_time = 0
        self.transport_count = 0
        self.last_state_change_time = 0
    
    def distance_to_station(self, station):
        """Calculate distance from current position to given station."""
        return DISTANCES[self.current_station-1][station-1]
    
    def assign_transport(self, from_station, to_station, current_time):
        """Assign transport task to forklift."""
        # Calculate travel time to pickup, then to destination (convert seconds to hours)
        travel_to_pickup = DISTANCES[self.current_station-1][from_station-1] / FORKLIFT_SPEED / 3600
        travel_to_dest = DISTANCES[from_station-1][to_station-1] / FORKLIFT_SPEED / 3600
        
        total_time = travel_to_pickup + travel_to_dest
        self.time_until_idle = total_time
        
        # Update statistics - track empty vs loaded movement
        if self.is_idle:
            self.total_idle_time += current_time - self.last_state_change_time
        
        # Empty movement (to pickup) and loaded movement (to destination)
        self.total_empty_time += travel_to_pickup
        self.total_loaded_time += travel_to_dest
        
        self.log('assigned transport', current_time, 
                f"from station {self.current_station} → pickup at {from_station} → dropoff at {to_station}")
        
        self.current_station = to_station
        self.is_idle = False
        self.transport_count += 1
        self.last_state_change_time = current_time
        
        return current_time + total_time

    def become_idle(self, current_time):
        """Mark forklift as idle and update statistics."""
        if not self.is_idle:
            self.total_transport_time += current_time - self.last_state_change_time
            self.is_idle = True
            self.time_until_idle = 0
            self.last_state_change_time = current_time
            self.log('became idle', current_time, f"at station {self.current_station}")

    def log(self, event, time, extra=None):
        msg = f"[t={time:.3f}] Forklift {self.forklift_id} - {event}"
        if extra:
            msg += f" | {extra}"
        self.trace.append(msg)
        # print(msg)  # Comment out for less verbose output
    
    def get_utilization(self, total_time):
        """Calculate forklift utilization statistics."""
        if total_time == 0:
            return {
                'total_utilization': 0,
                'loaded_utilization': 0,
                'empty_utilization': 0,
                'idle_utilization': 0
            }
        return {
            'total_utilization': self.total_transport_time / total_time,
            'loaded_utilization': self.total_loaded_time / total_time,
            'empty_utilization': self.total_empty_time / total_time,
            'idle_utilization': self.total_idle_time / total_time
        }

def assign_forklift(forklifts, from_station):
    idle_forklifts = [f for f in forklifts if f.is_idle]
    if not idle_forklifts:
        return None
    # Find the closest
    closest = min(idle_forklifts, key=lambda f: DISTANCES[f.current_station-1][from_station-1])
    return closest
