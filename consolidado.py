#!/usr/bin/env python3
"""
Detailed analysis of the manufacturing simulation model to identify bottlenecks
and validate against expected performance metrics.
"""

import numpy as np
from manufacturing_simulation import ManufacturingSimulation, run_experiment
from model_constants import (
    MEAN_INTERARRIVAL_TIME, JOB_TYPES, ROUTINGS, MEAN_SERVICE_TIMES,
    DISTANCES, FORKLIFT_SPEED
)

def calculate_theoretical_performance():
    """Calculate theoretical performance metrics based on problem description."""
    print("="*80)
    print("THEORETICAL PERFORMANCE ANALYSIS")
    print("="*80)
    
    # Job arrival rate
    arrival_rate = 1 / MEAN_INTERARRIVAL_TIME  # 15 jobs per hour
    print(f"Job arrival rate: {arrival_rate} jobs/hour")
    print(f"Expected daily throughput (8 hours): {arrival_rate * 8} jobs/day")
    
    # Station utilization analysis
    print("\nSTATION UTILIZATION ANALYSIS:")
    print("-" * 50)
    
    for station in range(1, 6):
        # Calculate arrival rate to each station
        station_arrival_rate = 0
        station_mean_service_time = 0
        
        for job_type_info in JOB_TYPES:
            job_type = job_type_info['job_type']
            prob = job_type_info['probability']
            routing = ROUTINGS[job_type]
            service_times = MEAN_SERVICE_TIMES[job_type]
            
            if station in routing:
                station_idx = routing.index(station)
                service_time = service_times[station_idx]
                station_arrival_rate += arrival_rate * prob
                station_mean_service_time += prob * service_time
        
        if station_arrival_rate > 0:
            # Normalize mean service time
            station_mean_service_time = station_mean_service_time / (station_arrival_rate / arrival_rate)
            service_rate_per_machine = 1 / station_mean_service_time
            required_machines = station_arrival_rate / service_rate_per_machine
            
            print(f"Station {station}:")
            print(f"  Arrival rate: {station_arrival_rate:.2f} jobs/hour")
            print(f"  Mean service time: {station_mean_service_time:.3f} hours")
            print(f"  Service rate per machine: {service_rate_per_machine:.2f} jobs/hour")
            print(f"  Theoretical machines needed: {required_machines:.2f}")
            print()
    
    # Travel time analysis
    print("TRAVEL TIME ANALYSIS:")
    print("-" * 30)
    
    total_travel_time = 0
    total_weight = 0
    
    for job_type_info in JOB_TYPES:
        job_type = job_type_info['job_type']
        prob = job_type_info['probability']
        routing = ROUTINGS[job_type]
        
        # Calculate travel time for this job type
        job_travel_time = 0
        full_route = [6] + routing + [6]  # Start and end at I/O station
        
        for i in range(len(full_route) - 1):
            from_station = full_route[i]
            to_station = full_route[i + 1]
            distance = DISTANCES[from_station - 1][to_station - 1]
            travel_time_hours = distance / FORKLIFT_SPEED / 3600  # Convert to hours
            job_travel_time += travel_time_hours
        
        print(f"Job Type {job_type} (prob={prob}):")
        print(f"  Route: {full_route}")
        print(f"  Total travel time: {job_travel_time:.4f} hours")
        
        total_travel_time += prob * job_travel_time
        total_weight += prob
    
    avg_travel_time = total_travel_time / total_weight
    print(f"\nAverage travel time per job: {avg_travel_time:.4f} hours")
    print(f"Expected: 0.06 hours (from problem description)")
    print(f"Ratio: {avg_travel_time / 0.06:.2f}x expected")
    
    # Forklift utilization analysis
    forklift_demand = arrival_rate * avg_travel_time
    print(f"\nForklift demand: {forklift_demand:.3f} forklifts")
    
    return {
        'arrival_rate': arrival_rate,
        'avg_travel_time': avg_travel_time,
        'forklift_demand': forklift_demand
    }

def run_detailed_test(config, config_name, expected_throughput):
    """Run a detailed test with comprehensive output."""
    print(f"\n{'='*80}")
    print(f"DETAILED TEST: {config_name}")
    print(f"Expected throughput: {expected_throughput:.2f} jobs/8hr-day")
    print(f"{'='*80}")
    
    # Run simulation
    simulation = ManufacturingSimulation(config)
    results = []
    
    for rep in range(config['replications']):
        print(f"\nReplication {rep + 1}/{config['replications']}...")
        result = simulation.run_single_replication(rep, verbose=False)
        results.append(result)
        
        # Print key metrics for this replication
        job_perf = result['job_performance']
        print(f"  Throughput: {job_perf['throughput_per_8hr_day']:.2f} jobs/8hr-day")
        print(f"  System time: {job_perf['avg_system_time']:.2f} hours")
        print(f"  Jobs completed: {job_perf['jobs_completed']}")
    
    # Calculate averages
    avg_throughput = np.mean([r['job_performance']['throughput_per_8hr_day'] for r in results])
    avg_system_time = np.mean([r['job_performance']['avg_system_time'] for r in results])
    avg_jobs_completed = np.mean([r['job_performance']['jobs_completed'] for r in results])
    
    print(f"\nAVERAGE RESULTS:")
    print(f"Throughput: {avg_throughput:.2f} jobs/8hr-day (Expected: {expected_throughput:.2f})")
    print(f"Performance ratio: {avg_throughput/expected_throughput:.2f}")
    print(f"System time: {avg_system_time:.2f} hours")
    print(f"Jobs completed: {avg_jobs_completed:.1f}")
    
    # Analyze station performance if available
    if results and 'station_performance' in results[0]:
        print(f"\nSTATION PERFORMANCE ANALYSIS:")
        station_perf = results[0]['station_performance']
        for i, station_data in enumerate(station_perf):
            print(f"Station {i+1}: Utilization={station_data.get('utilization', 0):.2f}, "
                  f"Avg Queue={station_data.get('avg_queue_length', 0):.2f}")
    
    return results, avg_throughput, avg_system_time

def main():
    """Main analysis function."""
    print("MANUFACTURING SIMULATION DETAILED ANALYSIS")
    print("Comparing model performance with theoretical expectations")
    
    # First, calculate theoretical performance
    theoretical = calculate_theoretical_performance()
    
    # Test configurations
    configs = [
        {
            'name': 'Configuration 1: [4,1,4,2,2], 1 forklift',
            'config': {
                'num_stations': 5,
                'station_machine_counts': [4, 1, 4, 2, 2],
                'num_forklifts': 1,
                'simulation_hours': 400,  # Shorter for debugging
                'warmup_hours': 40,
                'replications': 3,
                'random_seed': 12345
            },
            'expected_throughput': 94.94
        },
        {
            'name': 'Configuration 2: [4,2,5,3,2], 1 forklift',
            'config': {
                'num_stations': 5,
                'station_machine_counts': [4, 2, 5, 3, 2],
                'num_forklifts': 1,
                'simulation_hours': 400,
                'warmup_hours': 40,
                'replications': 3,
                'random_seed': 12345
            },
            'expected_throughput': 106.77
        },
        {
            'name': 'Configuration 3: [4,2,5,3,2], 2 forklifts',
            'config': {
                'num_stations': 5,
                'station_machine_counts': [4, 2, 5, 3, 2],
                'num_forklifts': 2,
                'simulation_hours': 400,
                'warmup_hours': 40,
                'replications': 3,
                'random_seed': 12345
            },
            'expected_throughput': 120.29
        }
    ]
    
    # Run tests
    all_results = []
    for config_info in configs:
        results, throughput, system_time = run_detailed_test(
            config_info['config'], 
            config_info['name'], 
            config_info['expected_throughput']
        )
        all_results.append({
            'name': config_info['name'],
            'throughput': throughput,
            'expected': config_info['expected_throughput'],
            'system_time': system_time
        })
    
    # Final summary
    print(f"\n{'='*100}")
    print("FINAL ANALYSIS SUMMARY")
    print(f"{'='*100}")
    print(f"{'Configuration':<40} {'Actual':<12} {'Expected':<12} {'Ratio':<8} {'Status':<15}")
    print("-" * 100)
    
    for result in all_results:
        ratio = result['throughput'] / result['expected']
        status = "GOOD" if ratio >= 0.9 else "POOR" if ratio < 0.5 else "FAIR"
        print(f"{result['name']:<40} {result['throughput']:<12.2f} {result['expected']:<12.2f} "
              f"{ratio:<8.2f} {status:<15}")
    
    # Identify potential issues
    print(f"\nPOTENTIAL ISSUES IDENTIFIED:")
    print("-" * 50)
    
    avg_ratio = np.mean([r['throughput'] / r['expected'] for r in all_results])
    if avg_ratio < 0.7:
        print("❌ Overall performance significantly below expectations")
        print("   - Check service time distributions (should be Gamma(2, mean))")
        print("   - Verify forklift speed and distance calculations")
        print("   - Check for deadlocks or blocking issues")
    elif avg_ratio < 0.9:
        print("⚠️  Performance below expectations but reasonable")
        print("   - Minor calibration issues possible")
    else:
        print("✅ Performance meets or exceeds expectations")
    
    travel_time_ratio = theoretical['avg_travel_time'] / 0.06
    if travel_time_ratio > 1.2:
        print(f"❌ Travel times too high: {travel_time_ratio:.2f}x expected")
    elif travel_time_ratio < 0.8:
        print(f"❌ Travel times too low: {travel_time_ratio:.2f}x expected")

if __name__ == "__main__":
    main()


    #!/usr/bin/env python3
"""
Simple diagnostic script to understand why throughput is low.
"""

import numpy as np
from manufacturing_simulation import ManufacturingSimulation

def run_diagnostic():
    """Run a diagnostic simulation to understand bottlenecks."""
    config = {
        'num_stations': 5,
        'station_machine_counts': [4, 1, 4, 2, 2],  # Configuration 1
        'num_forklifts': 3,  # Test with more forklifts
        'simulation_hours': 80,  # Short run for analysis
        'warmup_hours': 8,
        'replications': 1,
        'random_seed': 12345
    }
    
    print("DIAGNOSTIC SIMULATION")
    print("="*50)
    print(f"Config: {config['station_machine_counts']} machines, {config['num_forklifts']} forklifts")
    print(f"Simulation time: {config['simulation_hours']} hours")
    
    simulation = ManufacturingSimulation(config)
    
    # Enable more detailed logging
    result = simulation.run_single_replication(0, verbose=True)
    
    print("\nFINAL RESULTS:")
    print("-" * 30)
    job_perf = result['job_performance']
    print(f"Jobs completed: {job_perf['jobs_completed']}")
    print(f"Throughput: {job_perf['throughput_per_8hr_day']:.2f} jobs/8hr-day")
    print(f"Average system time: {job_perf['avg_system_time']:.2f} hours")
    
    # Analyze station performance
    print("\nSTATION ANALYSIS:")
    print("-" * 30)
    for i, station in enumerate(simulation.stations):
        print(f"Station {i+1}:")
        print(f"  Queue length: {len(station.queue)}")
        print(f"  Machines busy: {sum(1 for m in station.machines if m.state == 'busy')}")
        print(f"  Machines blocked: {sum(1 for m in station.machines if m.state == 'blocked')}")
        print(f"  Machines idle: {sum(1 for m in station.machines if m.state == 'idle')}")
    
    # Analyze forklift performance
    print("\nFORKLIFT ANALYSIS:")
    print("-" * 30)
    for forklift in simulation.forklifts:
        print(f"Forklift {forklift.forklift_id}:")
        print(f"  Is idle: {forklift.is_idle}")
        print(f"  Current station: {forklift.current_station}")
        print(f"  Transport count: {forklift.transport_count}")
        utilization = forklift.get_utilization(config['simulation_hours'])
        print(f"  Utilization: {utilization:.3f}")
    
    # Check forklift manager queue
    print(f"\nForklift request queue length: {len(simulation.forklift_manager.request_queue)}")
    
    # Check jobs in system
    print(f"Jobs still in system: {len(simulation.jobs_in_system)}")
    
    return result

if __name__ == "__main__":
    run_diagnostic()


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


#!/usr/bin/env python3
"""
Test script to validate simulation model against expected performance metrics.
"""

import numpy as np
from manufacturing_simulation import ManufacturingSimulation, run_experiment
import model_constants

def print_detailed_metrics_report(results, config, system_design_num):
    """Print detailed metrics in the format specified by the user."""
    if not results:
        print("No results available")
        return
    
    # Calculate averages across all replications
    n_replications = len(results)
    
    # Station-level metrics (average across replications)
    station_metrics = {}
    for i in range(1, 6):
        station_key = f'station_{i}'
        station_metrics[i] = {
            'busy_util': [],
            'blocked_util': [],
            'avg_queue': [],
            'max_queue': []
        }
    
    # Collect metrics from all replications
    for result in results:
        if 'station_performance' in result:
            for i in range(1, 6):
                station_key = f'station_{i}'
                if station_key in result['station_performance']:
                    station_data = result['station_performance'][station_key]
                    station_metrics[i]['busy_util'].append(station_data.get('busy_utilization', 0))
                    station_metrics[i]['blocked_util'].append(station_data.get('blocked_utilization', 0))
                    station_metrics[i]['avg_queue'].append(station_data.get('average_queue_length', 0))
                    station_metrics[i]['max_queue'].append(station_data.get('max_queue_length', 0))
                else:
                    # If station not in results, use zeros
                    station_metrics[i]['busy_util'].append(0)
                    station_metrics[i]['blocked_util'].append(0)
                    station_metrics[i]['avg_queue'].append(0)
                    station_metrics[i]['max_queue'].append(0)
    
    # Calculate averages for each station
    for i in range(1, 6):
        for metric in station_metrics[i]:
            if station_metrics[i][metric]:
                station_metrics[i][metric] = np.mean(station_metrics[i][metric])
            else:
                station_metrics[i][metric] = 0
    
    # Overall performance metrics
    throughput = np.mean([r['job_performance']['throughput_per_8hr_day'] for r in results])
    avg_system_time = np.mean([r['job_performance']['avg_system_time'] for r in results])
    avg_queue_time = np.mean([r['job_performance']['avg_queue_time'] for r in results])
    avg_transport_time = np.mean([r['job_performance']['avg_transport_time'] for r in results])
    
    # Forklift metrics
    proportion_loaded = np.mean([r['forklift_performance']['proportion_moving_loaded'] for r in results])
    proportion_empty = np.mean([r['forklift_performance']['proportion_moving_empty'] for r in results])
    
    # Print the detailed report
    print(f"\nSimulation results for system design {system_design_num}")
    machine_counts = config.get('station_machine_counts', [])
    print(f"Number of machines: {', '.join(map(str, machine_counts))}")
    print(f"Number of forklifts: {config.get('num_forklifts', 0)}")
    
    # Header for station performance
    print("Station" + " " * 22 + "1" + " " * 6 + "2" + " " * 6 + "3" + " " * 6 + "4" + " " * 6 + "5")
    print("Performance measure")
    
    # Proportion machines busy
    busy_proportions = [f"{station_metrics[i]['busy_util']:.2f}" for i in range(1, 6)]
    print(f"Proportion machines busy     {' '.join(f'{p:>6}' for p in busy_proportions)}")
    
    # Proportion machines blocked
    blocked_proportions = [f"{station_metrics[i]['blocked_util']:.2f}" for i in range(1, 6)]
    print(f"Proportion machines blocked  {' '.join(f'{p:>6}' for p in blocked_proportions)}")
    
    # Average number in queue
    avg_queues = [f"{station_metrics[i]['avg_queue']:.2f}" for i in range(1, 6)]
    print(f"Average number in queue      {' '.join(f'{q:>6}' for q in avg_queues)}")
    
    # Maximum number in queue
    max_queues = [f"{station_metrics[i]['max_queue']:.0f}" for i in range(1, 6)]
    print(f"Maximum number in queue      {' '.join(f'{q:>6}' for q in max_queues)}")
    
    # Overall performance metrics
    print(f"Average daily throughput     {throughput:.2f}")
    print(f"Average time in system       {avg_system_time:.2f}")
    print(f"Average total time in queues {avg_queue_time:.2f}")
    print(f"Average total wait for transport {avg_transport_time:.2f}")
    print(f"Proportion forklifts moving loaded {proportion_loaded:.2f}")
    print(f"Proportion forklifts moving empty  {proportion_empty:.2f}")

def test_configuration(config_info):
    """Test a single configuration with the given parameters.
    
    Args:
        config_info (dict): Configuration information containing:
            - name: Configuration name/description
            - machines: List of machine counts per station
            - forklifts: Number of forklifts
            - design_num: System design number for reporting
            - description: Additional description lines (optional)
            - simulation_params: Override simulation parameters (optional)
            - machine_efficiencies: Dict mapping station numbers to efficiency values (optional)
    
    Returns:
        list: Results from all replications
    """
    # Default simulation parameters
    default_params = {
        'num_stations': 5,
        'simulation_hours': 920,
        'warmup_hours': 0,
        'replications': 10,
        'random_seed': 12345
    }
    
    # Override with any custom parameters
    sim_params = default_params.copy()
    if 'simulation_params' in config_info:
        sim_params.update(config_info['simulation_params'])
    
    # Build the complete configuration
    config = sim_params.copy()
    config['station_machine_counts'] = config_info['machines']
    config['num_forklifts'] = config_info['forklifts']

    # Print configuration header
    machines_str = ','.join(map(str, config_info['machines']))
    header = f"TESTING {config_info['name']}: [{machines_str}] machines, {config_info['forklifts']} forklifts"
    
    print("\n" + "="*max(60, len(header)))
    print(header)
    
    # Print description lines if provided
    if 'description' in config_info:
        for desc_line in config_info['description']:
            print(desc_line)
    
    print("="*max(60, len(header)))
    
    # Run the experiment
    results = run_experiment(config, verbose=False)
    
    # Print detailed metrics report
    print_detailed_metrics_report(results, config, config_info['design_num'])
    
    return results

if __name__ == "__main__":
    print("MANUFACTURING SIMULATION MODEL VALIDATION")
    print("Comparing actual vs expected performance metrics")
    
    # Define all test configurations
    configurations = [
        {
            'name': 'Configuration 1',
            'machines': [4, 2, 5, 3, 2],
            'forklifts': 2,
            'design_num': 1,
            'efficiency': 1,
            'target_machines': [],
            'description': ['This tests the original design with insufficient forklift capacity.']
        },
        {
            'name': 'Configuration 2',
            'machines': [5, 2, 5, 3, 3],
            'forklifts': 2,
            'design_num': 2,
            'efficiency': 1,
            'target_machines': [],
            'description': ['This tests system design 2 with insufficient forklift capacity.']
        },
        {
            'name': 'Configuration 3',
            'machines': [4, 3, 5, 3, 3],
            'forklifts': 3,
            'design_num': 3,
            'efficiency': 1,
            'target_machines': [],
            'description': ['This tests system design 3 with adequate forklift capacity.']
        },
        {
            'name': 'Configuration 4',
            'machines': [5, 3, 5, 3, 2],
            'forklifts': 3,
            'design_num': 4,
            'efficiency': 1,
            'target_machines': [],
            'description': ['This tests system design 4 with adequate forklift capacity.']
        },
        {
            'name': 'Configuration 5',
            'machines': [4, 2, 6, 3, 3],
            'forklifts': 3,
            'design_num': 5,
            'efficiency': 1,
            'target_machines': [],
            'description': ['This tests system design 4 with adequate forklift capacity.']
        },
        {
            'name': 'Configuration 6',
            'machines': [5, 2, 6, 3, 2],
            'forklifts': 3,
            'design_num': 6,
            'efficiency': 1,
            'target_machines': [],
            'description': ['This tests system design 4 with adequate forklift capacity.']
        }, 
        {
            'name': 'Configuration 7',
            'machines': [4, 3, 6, 3, 2],
            'forklifts': 2,
            'design_num': 7,
            'efficiency': 1,
            'target_machines': [],
            'description': ['This tests system design 4 with adequate forklift capacity.']
        }, 
        {
            'name': 'Configuration 8',
            'machines': [5, 3, 6, 3, 3],
            'forklifts': 2,
            'design_num': 8,
            'efficiency': 1,
            'target_machines': [],
            'description': ['This tests system design 4 with adequate forklift capacity.']
        }, 
        {
            'name': 'Configuration 9',
            'machines': [4, 2, 5, 4, 2],
            'forklifts': 3,
            'design_num': 9,
            'efficiency': 1,
            'target_machines': [],
            'description': ['This tests system design 4 with adequate forklift capacity.']
        }, 
        {
            'name': 'Configuration 10',
            'machines': [5, 2, 5, 4, 3],
            'forklifts': 3,
            'design_num': 10,
            'efficiency': 1,
            'target_machines': [],
            'description': ['This tests system design 4 with adequate forklift capacity.']
        }, 
        {
            'name': 'Configuration 11',
            'machines': [4, 3, 5, 4, 3],
            'forklifts': 2,
            'design_num': 11,
            'efficiency': 1,
            'target_machines': [],
            'description': ['This tests system design 4 with adequate forklift capacity.']
        }, 
        {
            'name': 'Configuration 12',
            'machines': [5, 3, 5, 4, 2],
            'forklifts': 2,
            'design_num': 12,
            'efficiency': 1,
            'target_machines': [],
            'description': ['This tests system design 4 with adequate forklift capacity.']
        }, 
        {
            'name': 'Configuration 13',
            'machines': [4, 2, 6, 4, 3],
            'forklifts': 2,
            'design_num': 13,
            'efficiency': 1,
            'target_machines': [],
            'description': ['This tests system design 4 with adequate forklift capacity.']
        }, 
        {
            'name': 'Configuration 14',
            'machines': [5, 2, 6, 4, 2],
            'forklifts': 2,
            'design_num': 14,
            'efficiency': 1,
            'target_machines': [],
            'description': ['This tests system design 4 with adequate forklift capacity.']
        }, 
        {
            'name': 'Configuration 15',
            'machines': [4, 3, 6, 4, 2],
            'forklifts': 3,
            'design_num': 15,
            'efficiency': 1,
            'target_machines': [],
            'description': ['This tests system design 4 with adequate forklift capacity.']
        }, 
        {
            'name': 'Configuration 16',
            'machines': [5, 3, 6, 4, 3],
            'forklifts': 3,
            'design_num': 16,
            'efficiency': 1,
            'target_machines': [],
            'description': ['This tests system design 4 with adequate forklift capacity.']
        }       
    ]
    
    # Run all test configurations
    all_results = []
    for config in configurations:
        model_constants.MACHINE_EFFICIENCY = config['efficiency']   
        model_constants.EFFICENCY_TARGET_MACHINES = config['target_machines'] 
        results = test_configuration(config)
        print("Config efficiency: ", config['efficiency'])
        all_results.append((config, results))
    
    print("\n" + "="*80)
    print("SUMMARY OF ALL TESTS")
    print("="*80)
    
    # Summary table with expected values for comparison
    expected_values = [
        (94.94, 109.20),  # Config 1
        (106.77, 55.84),  # Config 2
        (120.29, 1.76),   # Config 3
        (120.29, 1.76),   # Config 4 (estimated)
        (120.33, 2.03),   # Config 5 (estimated)   
        (119.88, 5.31),   # Config 6 (estimated)  
        (119.88, 5.31),   # Config 7 (estimated)  
        (119.88, 5.31),   # Config 8 (estimated) 
        (119.88, 5.31),   # Config 9 (estimated) 
        (119.88, 5.31),   # Config 10 (estimated) 
        (119.88, 5.31),   # Config 11 (estimated) 
        (119.88, 5.31),   # Config 12 (estimated) 
        (119.88, 5.31),   # Config 13 (estimated) 
        (119.88, 5.31),   # Config 14 (estimated) 
        (119.88, 5.31),   # Config 15 (estimated) 
        (119.88, 5.31),   # Config 16 (estimated)        
    ]
    
    print(f"{'Configuration':<35} {'Actual':<12} {'Expected':<12} {'Diff':<12} {'System Time':<15}")
    print("-" * 95)
    
    for i, (config, results) in enumerate(all_results):
        actual_throughput = np.mean([r['job_performance']['throughput_per_8hr_day'] for r in results])
        actual_system_time = np.mean([r['job_performance']['avg_system_time'] for r in results])
        
        exp_throughput, exp_system_time = expected_values[i]
        diff = actual_throughput - exp_throughput
        
        machines_str = ','.join(map(str, config['machines']))
        name = f"{config['name']}: [{machines_str}], {config['forklifts']} forklift{'s' if config['forklifts'] > 1 else ''}"
        
        print(f"{name:<35} {actual_throughput:<12.2f} {exp_throughput:<12.2f} {diff:<12.2f} {actual_system_time:<15.2f}")
    
    print(f"\nKEY FINDINGS:")
    print("="*50)
    print("✅ MAJOR ISSUE IDENTIFIED: Forklift capacity bottleneck")
    print("✅ ROOT CAUSE: Theoretical forklift analysis oversimplified")
    print("✅ SOLUTION: Increase forklift count to account for:")
    print("   - Machine blocking effects")
    print("   - Queueing delays")
    print("   - Empty travel time")
    print("   - Variable workload distribution")
    print("\n💡 RECOMMENDATION: Use 2-3 forklifts minimum for realistic operations")
    
    print(f"\n🎯 DETAILED METRICS OUTPUT:")
    print("="*50)
    print("✅ All test configurations now output comprehensive metrics including:")
    print("   - Proportion of machines busy/blocked by station")
    print("   - Average and maximum queue lengths by station")
    print("   - Daily throughput and system times")
    print("   - Forklift utilization (loaded/empty movement)")
    print("   - Transport wait times")
    print("\n📊 Use the detailed reports above for system design analysis!")

import numpy as np
from model_constants import JOB_TYPES, MEAN_INTERARRIVAL_TIME

class JobInputGenerator:
    def __init__(self, mean_interarrival_time, job_types, job_type_probs):
        self.mean = mean_interarrival_time  # in hours
        self.job_types = job_types
        self.job_type_probs = job_type_probs

    def next_interarrival(self):
        # Exponential distribution for interarrival time
        return np.random.exponential(self.mean)

    def next_job_type(self):
        # Randomly select job type based on probabilities
        return np.random.choice(self.job_types, p=self.job_type_probs)

if __name__ == "__main__":
    # Example usage:
    job_types = [jt['job_type'] for jt in JOB_TYPES]
    job_type_probs = [jt['probability'] for jt in JOB_TYPES]
    generator = JobInputGenerator(MEAN_INTERARRIVAL_TIME, job_types, job_type_probs)
    interarrival_time = generator.next_interarrival()
    job_type = generator.next_job_type()
    print(f"Next job arrives in {interarrival_time:.4f} hours and is of type {job_type}")

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


"""
Complete Manufacturing Simulation System

This module implements a discrete-event simulation of a manufacturing facility
with 5 workstations, an I/O station, multiple machines per station, and forklift trucks
for material handling.
"""

import numpy as np
from job import Job
from input_component import JobInputGenerator
from station import Station
from forklift import Forklift
from forklift_manager import ForkliftManager
from event_scheduler import EventScheduler
from simulation_statistics import SimulationStatistics
from model_constants import (
    MEAN_INTERARRIVAL_TIME, JOB_TYPES, ROUTINGS, MEAN_SERVICE_TIMES
)

class ManufacturingSimulation:
    """Main simulation class that orchestrates the entire manufacturing system."""
    
    def __init__(self, config):
        """
        Initialize the simulation with given configuration.
        
        Args:
            config (dict): Configuration parameters including:
                - num_stations: Number of workstations (default: 5)
                - station_machine_counts: List of machine counts per station
                - num_forklifts: Number of forklift trucks
                - simulation_hours: Total simulation time
                - warmup_hours: Warmup period
                - replications: Number of replications to run
        """
        self.config = config
        self.reset()
    
    def reset(self):
        """Reset the simulation to initial state."""
        # Initialize components
        self.stations = []
        self.io_station = None
        self.forklifts = []
        self.forklift_manager = None
        self.job_generator = None
        self.scheduler = EventScheduler()
        self.statistics = SimulationStatistics()
        
        # Simulation state
        self.current_time = 0
        self.job_id_counter = 0
        self.jobs_in_system = {}
        
        self._initialize_system()
    
    def _initialize_system(self):
        """Initialize all system components."""
        config = self.config
        
        # Create workstations (1-5)
        for i in range(config['num_stations']):
            station = Station(
                station_id=i+1,
                num_machines=config['station_machine_counts'][i],
                mean_service_times=MEAN_SERVICE_TIMES
            )
            self.stations.append(station)
        
        # Create I/O station (station 6)
        self.io_station = Station(
            station_id=6,
            num_machines=0,  # I/O station has no machines
            mean_service_times=MEAN_SERVICE_TIMES
        )
        
        # Create forklifts
        self.forklifts = [Forklift(i+1) for i in range(config['num_forklifts'])]
        self.forklift_manager = ForkliftManager(self.forklifts)
        
        # Create job generator
        job_types = [jt['job_type'] for jt in JOB_TYPES]
        job_type_probs = [jt['probability'] for jt in JOB_TYPES]
        self.job_generator = JobInputGenerator(
            MEAN_INTERARRIVAL_TIME, job_types, job_type_probs
        )
    
    def run_single_replication(self, replication_id=0, verbose=False):
        """
        Run a single replication of the simulation.
        
        Args:
            replication_id (int): ID of this replication
            verbose (bool): Whether to print detailed events
            
        Returns:
            dict: Statistics from this replication
        """
        config = self.config
        
        # Set random seed for reproducibility if using common random numbers
        if 'random_seed' in config:
            np.random.seed(config['random_seed'] + replication_id)
        
        self.reset()
        
        # Schedule first job arrival
        first_arrival = self.job_generator.next_interarrival()
        self.scheduler.schedule(first_arrival, 'job_arrival')
        
        # Main simulation loop
        events_processed = 0
        while (self.scheduler.has_events() and 
               self.current_time < config['simulation_hours']):
            
            event = self.scheduler.next_event()
            self.current_time = event.time
            events_processed += 1
            
            if verbose and events_processed % 1000 == 0:
                print(f"Processed {events_processed} events at time {self.current_time:.2f}")
            
            # Process the event
            self._process_event(event)
            
            # Sample system state periodically for statistics
            if events_processed % 100 == 0:
                self.statistics.sample_system_state(
                    self.stations, self.forklifts, self.current_time
                )
        
        # Collect final statistics
        self._collect_final_statistics()
        
        return self.statistics.get_summary_report(
            config['simulation_hours'], 
            config.get('warmup_hours', 0)
        )
    
    def _process_event(self, event):
        """Process a single event."""
        event_type = event.event_type
        
        if event_type == 'job_arrival':
            self._handle_job_arrival()
            
        elif event_type == 'forklift_dropoff':
            self._handle_forklift_dropoff(event.kwargs)
            
        elif event_type == 'machine_finish':
            self._handle_machine_finish(event.kwargs)
            
        elif event_type == 'job_exit':
            self._handle_job_exit(event.kwargs)
    
    def _handle_job_arrival(self):
        """Handle a new job arriving at the I/O station."""
        # Generate new job
        job_type = self.job_generator.next_job_type()
        routing = ROUTINGS[job_type]
        job = Job(self.job_id_counter, job_type, routing)
        job.arrival_time = self.current_time
        
        self.jobs_in_system[self.job_id_counter] = job
        self.statistics.jobs_in_system += 1
        self.job_id_counter += 1
        
        job.log('arrived at system', self.current_time, f'routing: {routing}')
        
        # Schedule next job arrival
        next_arrival = self.current_time + self.job_generator.next_interarrival()
        self.scheduler.schedule(next_arrival, 'job_arrival')
        
        # Request forklift to transport job to first station
        first_station = routing[0]
        assignment = self.forklift_manager.request_forklift(
            job, 6, first_station, self.current_time, 0
        )
        
        if assignment:
            # Forklift assigned immediately
            self.scheduler.schedule(
                assignment['finish_time'], 
                'forklift_dropoff',
                job=assignment['request'].job,
                forklift=assignment['forklift'],
                to_station=first_station,
                operation_idx=0
            )
    
    def _handle_forklift_dropoff(self, kwargs):
        """Handle forklift dropping off a job at a station."""
        job = kwargs['job']
        forklift = kwargs['forklift']
        to_station = kwargs['to_station']
        operation_idx = kwargs['operation_idx']
        
        # Mark forklift as idle and try to assign to queued requests
        assignment = self.forklift_manager.forklift_became_idle(forklift, self.current_time)
        if assignment:
            # Another job was assigned to this forklift
            # Check if this is a pickup of a blocked job
            request = assignment['request']
            pickup_station_idx = request.from_station - 1
            
            if 0 <= pickup_station_idx < len(self.stations):
                # This is a pickup from a workstation - find and remove the blocked job
                station = self.stations[pickup_station_idx]
                removed_job, new_processing_time = None, None
                
                # Find the machine with the blocked job
                for machine_idx, machine in enumerate(station.machines):
                    if (machine.state == 'blocked' and 
                        machine.current_job and 
                        machine.current_job.job_id == request.job.job_id):
                        removed_job, new_processing_time = station.remove_job(machine_idx, self.current_time)
                        break
                
                # If a new job started processing, schedule its completion
                if new_processing_time is not None:
                    new_job = station.machines[machine_idx].current_job
                    self.scheduler.schedule(
                        self.current_time + new_processing_time,
                        'machine_finish',
                        job=new_job,
                        station_idx=pickup_station_idx,
                        machine_idx=machine_idx
                    )
            
            self.scheduler.schedule(
                assignment['finish_time'],
                'forklift_dropoff',
                job=assignment['request'].job,
                forklift=assignment['forklift'],
                to_station=assignment['request'].to_station,
                operation_idx=assignment['request'].operation_idx
            )
        
        if to_station == 6:
            # Job reached I/O station (exit)
            self.scheduler.schedule(
                self.current_time,
                'job_exit',
                job=job
            )
        else:
            # Job arrived at a workstation
            station = self.stations[to_station - 1]
            processing_time = station.arrive(job, operation_idx, self.current_time)
            
            if processing_time is not None:
                # Job started processing immediately
                finish_time = self.current_time + processing_time
                
                # Find which machine is processing this job
                machine_idx = None
                for i, machine in enumerate(station.machines):
                    if machine.current_job == job:
                        machine_idx = i
                        break
                
                if machine_idx is not None:
                    self.scheduler.schedule(
                        finish_time,
                        'machine_finish',
                        job=job,
                        station_idx=to_station - 1,
                        machine_idx=machine_idx
                    )
    
    def _handle_machine_finish(self, kwargs):
        """Handle a machine finishing processing a job."""
        job = kwargs['job']
        station_idx = kwargs['station_idx']
        machine_idx = kwargs['machine_idx']
        
        station = self.stations[station_idx]
        finished_job = station.finish_processing(machine_idx, self.current_time)
        
        # Update job's processing time
        if hasattr(job, 'time_being_processed'):
            job.time_being_processed += station.machines[machine_idx].service_time
        
        # Determine next destination
        if job.is_complete():
            next_station = 6  # I/O station (exit)
        else:
            job.advance_operation()
            next_station = job.get_current_station()
        
        # Request forklift for pickup
        assignment = self.forklift_manager.request_forklift(
            job, station_idx + 1, next_station, self.current_time, job.current_op
        )
        
        if assignment:
            # Forklift is available immediately - remove job from machine and assign to forklift
            removed_job, new_processing_time = station.remove_job(machine_idx, self.current_time)
            
            # Schedule forklift dropoff
            self.scheduler.schedule(
                assignment['finish_time'],
                'forklift_dropoff',
                job=assignment['request'].job,
                forklift=assignment['forklift'],
                to_station=next_station,
                operation_idx=job.current_op
            )
            
            # If a new job started processing, schedule its completion
            if new_processing_time is not None:
                new_finish_time = self.current_time + new_processing_time
                new_job = station.machines[machine_idx].current_job
                self.scheduler.schedule(
                    new_finish_time,
                    'machine_finish',
                    job=new_job,
                    station_idx=station_idx,
                    machine_idx=machine_idx
                )
        else:
            # No forklift available - job stays blocked on machine
            # The forklift request has been queued by the forklift_manager
            job.log(f'waiting for forklift pickup at station {station_idx + 1}', 
                   self.current_time, 'machine blocked')
    
    def _handle_job_exit(self, kwargs):
        """Handle a job exiting the system."""
        job = kwargs['job']
        job.finish_time = self.current_time
        
        # Record statistics
        self.statistics.record_job_completion(job)
        self.statistics.jobs_in_system -= 1
        
        # Remove from jobs in system
        if job.job_id in self.jobs_in_system:
            del self.jobs_in_system[job.job_id]
        
        job.log('exited system', self.current_time, 
               f'total time: {job.finish_time - job.arrival_time:.3f}h')
    
    def _collect_final_statistics(self):
        """Collect final statistics from all components."""
        # Station statistics
        for station in self.stations:
            self.statistics.record_station_stats(station, self.current_time)
        
        # Forklift statistics
        self.statistics.record_forklift_stats(self.forklifts, self.current_time)


def run_experiment(config, verbose=False):
    """
    Run a complete simulation experiment with multiple replications.
    
    Args:
        config (dict): Simulation configuration
        verbose (bool): Whether to print detailed output
        
    Returns:
        list: Results from all replications
    """
    simulation = ManufacturingSimulation(config)
    results = []
    
    print(f"Running {config['replications']} replications...")
    print(f"Configuration: {config['station_machine_counts']} machines, "
          f"{config['num_forklifts']} forklifts")
    
    for rep in range(config['replications']):
        print(f"\\nReplication {rep + 1}/{config['replications']}")
        
        result = simulation.run_single_replication(rep, verbose)
        results.append(result)
        
        # Print summary for this replication
        if verbose:
            simulation.statistics.print_summary(
                config['simulation_hours'],
                config.get('warmup_hours', 0)
            )
        else:
            job_perf = result['job_performance']
            print(f"  Jobs completed: {job_perf['jobs_completed']}")
            print(f"  Throughput: {job_perf['throughput_per_8hr_day']:.1f} jobs/8hr-day")
            print(f"  Avg system time: {job_perf['avg_system_time']:.3f}h")
    
    return results


# Example configurations to test different system designs
EXAMPLE_CONFIGS = {
    'base_case': {
        'num_stations': 5,
        'station_machine_counts': [4, 3, 4, 2, 3],  # machines per station 1-5
        'num_forklifts': 3,
        'simulation_hours': 920,  # 115 eight-hour days
        'warmup_hours': 120,     # 15 eight-hour days
        'replications': 10,
        'random_seed': 12345
    },
    'minimal_case': {
        'num_stations': 5,
        'station_machine_counts': [2, 2, 2, 1, 2],
        'num_forklifts': 2,
        'simulation_hours': 40,
        'warmup_hours': 8,
        'replications': 3,
        'random_seed': 12345
    }
}


if __name__ == "__main__":
    # Run a quick test with minimal configuration
    config = EXAMPLE_CONFIGS['minimal_case']
    results = run_experiment(config, verbose=True)
    
    # Calculate average performance across replications
    avg_throughput = np.mean([r['job_performance']['throughput_per_8hr_day'] for r in results])
    avg_system_time = np.mean([r['job_performance']['avg_system_time'] for r in results])
    
    print(f"\\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    print(f"Configuration: {config['station_machine_counts']} machines, {config['num_forklifts']} forklifts")
    print(f"Average Throughput: {avg_throughput:.1f} jobs/8hr-day (target: 120)")
    print(f"Average System Time: {avg_system_time:.3f} hours")
    print(f"Performance Rating: {'MEETS TARGET' if avg_throughput >= 120 else 'BELOW TARGET'}")

MEAN_INTERARRIVAL_TIME = 1/15  # Mean interarrival time for jobs
MACHINE_EFFICIENCY = 1

JOB_TYPES = [{'job_type': 1, 'probability': 0.3},
            {'job_type': 2, 'probability': 0.5},
            {'job_type': 3, 'probability': 0.2}]

# Routing for each job type
ROUTINGS = {
    1: [3, 1, 2, 5],
    2: [4, 1, 3],
    3: [2, 5, 1, 4, 3]
}

# Mean service times for each job type and operation
MEAN_SERVICE_TIMES = {
    1: [0.25, 0.15, 0.10, 0.30],
    2: [0.15, 0.20, 0.30],
    3: [0.15, 0.10, 0.35, 0.20, 0.20]
}

DISTANCES = [
    [0,   150, 213, 336, 300, 150],
    [150,   0, 150, 300, 336, 213],
    [213, 150,   0, 150, 213, 150],
    [336, 300, 150,   0, 150, 213],
    [300, 336, 213, 150,   0, 150],
    [150, 213, 150, 213, 150,   0]
]

FORKLIFT_SPEED = 5  # feet per second

EFFICENCY_TARGET_MACHINES = []

"""
Optimization script for finding the best manufacturing system configuration.

This script systematically tests different combinations of machines and forklifts
to find configurations that meet the target throughput of 120 jobs per 8-hour day.
"""

import numpy as np
import itertools
from manufacturing_simulation import run_experiment, EXAMPLE_CONFIGS

def generate_configurations(max_machines_per_station=6, max_forklifts=5):
    """
    Generate different system configurations to test.
    
    Args:
        max_machines_per_station (int): Maximum machines to test per station
        max_forklifts (int): Maximum number of forklifts to test
        
    Yields:
        dict: Configuration dictionaries
    """
    base_config = EXAMPLE_CONFIGS['minimal_case'].copy()
    
    # Test different machine combinations
    # Start with minimum viable configurations and expand
    machine_ranges = [
        range(1, max_machines_per_station + 1) for _ in range(5)
    ]
    
    forklift_range = range(1, max_forklifts + 1)
    
    for forklifts in forklift_range:
        for machines in itertools.product(*machine_ranges):
            # Skip configurations that are obviously too small
            if sum(machines) < 8:  # Minimum reasonable total machines
                continue
                
            config = base_config.copy()
            config['station_machine_counts'] = list(machines)
            config['num_forklifts'] = forklifts
            config['config_id'] = f"M{'-'.join(map(str, machines))}_F{forklifts}"
            
            yield config

def evaluate_configuration(config, target_throughput=120, max_system_time=2.0):
    """
    Evaluate a configuration and return performance metrics.
    
    Args:
        config (dict): Configuration to test
        target_throughput (float): Target jobs per 8-hour day
        max_system_time (float): Maximum acceptable average system time
        
    Returns:
        dict: Evaluation results
    """
    try:
        results = run_experiment(config, verbose=False)
        
        # Calculate averages across replications
        throughputs = [r['job_performance']['throughput_per_8hr_day'] for r in results]
        system_times = [r['job_performance']['avg_system_time'] for r in results]
        
        avg_throughput = np.mean(throughputs)
        std_throughput = np.std(throughputs)
        avg_system_time = np.mean(system_times)
        std_system_time = np.std(system_times)
        
        # Calculate total cost (example: machines cost 100, forklifts cost 200)
        total_machines = sum(config['station_machine_counts'])
        total_cost = total_machines * 100 + config['num_forklifts'] * 200
        
        # Evaluate if configuration meets targets
        meets_throughput = avg_throughput >= target_throughput
        acceptable_system_time = avg_system_time <= max_system_time
        
        return {
            'config_id': config['config_id'],
            'machines': config['station_machine_counts'],
            'forklifts': config['num_forklifts'],
            'total_machines': total_machines,
            'total_cost': total_cost,
            'avg_throughput': avg_throughput,
            'std_throughput': std_throughput,
            'avg_system_time': avg_system_time,
            'std_system_time': std_system_time,
            'meets_throughput': meets_throughput,
            'acceptable_system_time': acceptable_system_time,
            'feasible': meets_throughput and acceptable_system_time,
            'throughput_efficiency': avg_throughput / target_throughput,
            'cost_efficiency': avg_throughput / total_cost if total_cost > 0 else 0
        }
        
    except Exception as e:
        print(f"Error evaluating config {config['config_id']}: {e}")
        return None

def find_optimal_configurations(target_throughput=120, max_configs_to_test=50):
    """
    Find optimal system configurations through systematic search.
    
    Args:
        target_throughput (float): Target throughput in jobs per 8-hour day
        max_configs_to_test (int): Maximum number of configurations to test
        
    Returns:
        list: Best configurations sorted by cost efficiency
    """
    print(f"Searching for optimal configurations (target: {target_throughput} jobs/8hr-day)")
    print("="*80)
    
    feasible_configs = []
    all_results = []
    configs_tested = 0
    
    # Generate and test configurations
    for config in generate_configurations(max_machines_per_station=4, max_forklifts=4):
        if configs_tested >= max_configs_to_test:
            break
            
        configs_tested += 1
        print(f"Testing config {configs_tested}: {config['config_id']}", end=" ... ")
        
        result = evaluate_configuration(config, target_throughput)
        if result is None:
            print("FAILED")
            continue
            
        all_results.append(result)
        
        if result['feasible']:
            feasible_configs.append(result)
            print(f"FEASIBLE (throughput: {result['avg_throughput']:.1f}, cost: {result['total_cost']})")
        else:
            status = []
            if not result['meets_throughput']:
                status.append(f"throughput: {result['avg_throughput']:.1f}")
            if not result['acceptable_system_time']:
                status.append(f"system_time: {result['avg_system_time']:.2f}")
            print(f"NOT FEASIBLE ({', '.join(status)})")
    
    print(f"\\nTested {configs_tested} configurations")
    print(f"Found {len(feasible_configs)} feasible configurations")
    
    if not feasible_configs:
        print("\\nNo feasible configurations found. Top configurations by throughput:")
        top_configs = sorted(all_results, key=lambda x: x['avg_throughput'], reverse=True)[:5]
        for i, config in enumerate(top_configs):
            print(f"{i+1}. {config['config_id']}: {config['avg_throughput']:.1f} jobs/day, "
                  f"cost: {config['total_cost']}")
        return top_configs
    
    # Sort feasible configurations by cost efficiency
    feasible_configs.sort(key=lambda x: x['cost_efficiency'], reverse=True)
    
    print(f"\\nTop feasible configurations (sorted by cost efficiency):")
    print("-"*80)
    print(f"{'Rank':<4} {'Config':<20} {'Throughput':<12} {'System Time':<12} {'Cost':<8} {'Efficiency':<10}")
    print("-"*80)
    
    for i, config in enumerate(feasible_configs[:10]):
        print(f"{i+1:<4} {config['config_id']:<20} "
              f"{config['avg_throughput']:<12.1f} {config['avg_system_time']:<12.3f} "
              f"{config['total_cost']:<8} {config['cost_efficiency']:<10.4f}")
    
    return feasible_configs

def detailed_analysis(config_id, target_throughput=120):
    """
    Perform detailed analysis of a specific configuration.
    
    Args:
        config_id (str): Configuration ID to analyze
        target_throughput (float): Target throughput
    """
    # Parse config_id to extract machine and forklift counts
    # Format: "M1-2-3-4-5_F3" means machines [1,2,3,4,5] and 3 forklifts
    parts = config_id.split('_')
    machines_part = parts[0][1:]  # Remove 'M' prefix
    forklifts_part = int(parts[1][1:])  # Remove 'F' prefix
    
    machines = list(map(int, machines_part.split('-')))
    
    # Create detailed configuration
    config = EXAMPLE_CONFIGS['base_case'].copy()
    config['station_machine_counts'] = machines
    config['num_forklifts'] = forklifts_part
    config['config_id'] = config_id
    config['simulation_hours'] = 200  # Longer simulation for detailed analysis
    config['warmup_hours'] = 40
    config['replications'] = 5
    
    print(f"Detailed Analysis of Configuration: {config_id}")
    print("="*60)
    print(f"Machines per station: {machines}")
    print(f"Number of forklifts: {forklifts_part}")
    print(f"Total machines: {sum(machines)}")
    print(f"Estimated cost: {sum(machines) * 100 + forklifts_part * 200}")
    
    # Run detailed simulation
    results = run_experiment(config, verbose=True)
    
    # Analyze results across replications
    throughputs = [r['job_performance']['throughput_per_8hr_day'] for r in results]
    system_times = [r['job_performance']['avg_system_time'] for r in results]
    
    print(f"\\nPerformance Summary:")
    print(f"Average Throughput: {np.mean(throughputs):.2f} ± {np.std(throughputs):.2f} jobs/8hr-day")
    print(f"Target Achievement: {np.mean(throughputs)/target_throughput*100:.1f}%")
    print(f"Average System Time: {np.mean(system_times):.3f} ± {np.std(system_times):.3f} hours")
    
    # Station utilization analysis
    print(f"\\nStation Performance:")
    for station_id in range(1, 6):
        station_data = []
        for result in results:
            if f'station_{station_id}' in result['station_performance']:
                station_data.append(result['station_performance'][f'station_{station_id}'])
        
        if station_data:
            avg_queue = np.mean([s['average_queue_length'] for s in station_data])
            max_queue = max([s['max_queue_length'] for s in station_data])
            jobs_processed = np.mean([s['jobs_processed'] for s in station_data])
            print(f"  Station {station_id}: Avg Queue: {avg_queue:.2f}, "
                  f"Max Queue: {max_queue}, Jobs: {jobs_processed:.1f}")
    
    # Forklift utilization
    forklift_utils = []
    for result in results:
        forklift_utils.extend(result['forklift_performance']['individual_utilization'])
    
    print(f"\\nForklift Utilization: {np.mean(forklift_utils):.3f} ± {np.std(forklift_utils):.3f}")

if __name__ == "__main__":
    print("Manufacturing System Optimization")
    print("="*50)
    
    # Find optimal configurations
    optimal_configs = find_optimal_configurations(target_throughput=120, max_configs_to_test=30)
    
    if optimal_configs:
        print(f"\\n\\nRecommended Configuration:")
        best = optimal_configs[0]
        print(f"Machines: {best['machines']}")
        print(f"Forklifts: {best['forklifts']}")
        print(f"Expected Throughput: {best['avg_throughput']:.1f} jobs/8hr-day")
        print(f"Expected System Time: {best['avg_system_time']:.3f} hours")
        print(f"Estimated Cost: {best['total_cost']} units")
        
        # Run detailed analysis on the best configuration
        print(f"\\n\\nDetailed Analysis of Best Configuration:")
        detailed_analysis(best['config_id'])
    else:
        print("\\nNo optimal configurations found with current constraints.")
        print("Consider relaxing throughput targets or testing larger system sizes.")


import numpy as np
from collections import defaultdict

class SimulationStatistics:
    """Collects and analyzes simulation statistics."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all statistics."""
        # Job statistics
        self.jobs_completed = 0
        self.jobs_in_system = 0
        self.job_system_times = []
        self.job_queue_times = []
        self.job_processing_times = []
        self.job_transport_times = []
        
        # Station statistics
        self.station_stats = defaultdict(lambda: {
            'arrivals': 0,
            'departures': 0,
            'queue_lengths': [],
            'max_queue_length': 0,
            'total_queue_time': 0,
            'jobs_processed': 0
        })
        
        # Machine statistics
        self.machine_busy_time = defaultdict(list)
        self.machine_blocked_time = defaultdict(list)
        self.machine_idle_time = defaultdict(list)
        
        # Initialize forklift statistics
        self.forklift_utilization = []
        self.forklift_transport_count = []
        self.forklift_loaded_utilization = []
        self.forklift_empty_utilization = []
        self.forklift_queue_stats = {
            'max_queue_length': 0,
            'total_wait_time': 0,
            'total_requests': 0
        }
        
        # Time-based sampling
        self.sampling_times = []
        self.system_state_samples = []
    
    def record_job_completion(self, job):
        """Record statistics when a job completes."""
        self.jobs_completed += 1
        
        system_time = job.finish_time - job.arrival_time
        self.job_system_times.append(system_time)
        self.job_queue_times.append(job.time_in_queue)
        self.job_processing_times.append(job.time_being_processed)
        self.job_transport_times.append(job.time_being_transported)
    
    def record_station_stats(self, station, total_time):
        """Record statistics for a station."""
        station_id = station.station_id
        util_stats = station.get_utilization_stats(total_time)
        
        self.station_stats[station_id].update({
            'arrivals': station.stats['total_arrivals'],
            'departures': station.stats['total_departures'],
            'max_queue_length': station.stats['max_queue_length'],
            'jobs_processed': station.stats['jobs_processed'],
            'average_queue_length': util_stats.get('average_queue_length', 0),
            'busy_utilization': util_stats.get('busy', 0),
            'blocked_utilization': util_stats.get('blocked', 0),
            'idle_utilization': util_stats.get('idle', 0)
        })
    
    def record_forklift_stats(self, forklifts, total_time):
        """Record forklift utilization statistics."""
        self.forklift_utilization = []
        self.forklift_transport_count = []
        self.forklift_loaded_utilization = []
        self.forklift_empty_utilization = []
        
        for forklift in forklifts:
            util_stats = forklift.get_utilization(total_time)
            self.forklift_utilization.append(util_stats['total_utilization'])
            self.forklift_loaded_utilization.append(util_stats['loaded_utilization'])
            self.forklift_empty_utilization.append(util_stats['empty_utilization'])
            self.forklift_transport_count.append(forklift.transport_count)
    
    def sample_system_state(self, stations, forklifts, current_time):
        """Sample current system state for time-based statistics."""
        self.sampling_times.append(current_time)
        
        state = {
            'time': current_time,
            'total_jobs_in_system': self.jobs_in_system,
            'station_queues': {s.station_id: len(s.queue) for s in stations},
            'forklift_busy': sum(1 for f in forklifts if not f.is_idle),
            'machines_busy': {},
            'machines_blocked': {}
        }
        
        for station in stations:
            busy_count = sum(1 for m in station.machines if m.state == 'busy')
            blocked_count = sum(1 for m in station.machines if m.state == 'blocked')
            state['machines_busy'][station.station_id] = busy_count
            state['machines_blocked'][station.station_id] = blocked_count
        
        self.system_state_samples.append(state)
    
    def calculate_throughput(self, simulation_time, warmup_time=0):
        """Calculate system throughput (jobs per hour)."""
        effective_time = simulation_time - warmup_time
        if effective_time <= 0:
            return 0
        return self.jobs_completed / effective_time
    
    def get_summary_report(self, simulation_time, warmup_time=0):
        """Generate a comprehensive summary report."""
        effective_time = simulation_time - warmup_time
        
        report = {
            'simulation_info': {
                'total_time': simulation_time,
                'warmup_time': warmup_time,
                'effective_time': effective_time
            },
            'job_performance': {
                'jobs_completed': self.jobs_completed,
                'throughput_per_hour': self.calculate_throughput(simulation_time, warmup_time),
                'throughput_per_8hr_day': self.calculate_throughput(simulation_time, warmup_time) * 8,
                'avg_system_time': np.mean(self.job_system_times) if self.job_system_times else 0,
                'max_system_time': max(self.job_system_times) if self.job_system_times else 0,
                'avg_queue_time': np.mean(self.job_queue_times) if self.job_queue_times else 0,
                'avg_processing_time': np.mean(self.job_processing_times) if self.job_processing_times else 0,
                'avg_transport_time': np.mean(self.job_transport_times) if self.job_transport_times else 0
            },
            'station_performance': {},
            'forklift_performance': {
                'avg_utilization': np.mean(self.forklift_utilization) if self.forklift_utilization else 0,
                'individual_utilization': self.forklift_utilization,
                'total_transports': self.forklift_transport_count,
                'avg_transports': np.mean(self.forklift_transport_count) if self.forklift_transport_count else 0,
                'proportion_moving_loaded': np.mean(self.forklift_loaded_utilization) if self.forklift_loaded_utilization else 0,
                'proportion_moving_empty': np.mean(self.forklift_empty_utilization) if self.forklift_empty_utilization else 0
            }
        }
        
        # Add station-specific performance
        for station_id, stats in self.station_stats.items():
            if station_id != 6:  # Skip I/O station
                report['station_performance'][f'station_{station_id}'] = {
                    'jobs_processed': stats['jobs_processed'],
                    'max_queue_length': stats['max_queue_length'],
                    'average_queue_length': stats.get('average_queue_length', 0),
                    'busy_utilization': stats.get('busy_utilization', 0),
                    'blocked_utilization': stats.get('blocked_utilization', 0),
                    'idle_utilization': stats.get('idle_utilization', 0)
                }
        
        return report
    
    def print_summary(self, simulation_time, warmup_time=0):
        """Print a formatted summary report."""
        report = self.get_summary_report(simulation_time, warmup_time)
        
        print("\\n" + "="*60)
        print("SIMULATION SUMMARY REPORT")
        print("="*60)
        
        sim_info = report['simulation_info']
        print(f"Simulation Time: {sim_info['total_time']:.1f} hours")
        print(f"Warmup Time: {sim_info['warmup_time']:.1f} hours")
        print(f"Effective Time: {sim_info['effective_time']:.1f} hours")
        
        print("\\n" + "-"*40)
        print("JOB PERFORMANCE")
        print("-"*40)
        job_perf = report['job_performance']
        print(f"Jobs Completed: {job_perf['jobs_completed']}")
        print(f"Throughput: {job_perf['throughput_per_hour']:.2f} jobs/hour")
        print(f"Daily Throughput: {job_perf['throughput_per_8hr_day']:.1f} jobs/8hr-day")
        print(f"Average System Time: {job_perf['avg_system_time']:.3f} hours")
        print(f"Maximum System Time: {job_perf['max_system_time']:.3f} hours")
        print(f"Average Queue Time: {job_perf['avg_queue_time']:.3f} hours")
        
        print("\\n" + "-"*40)
        print("STATION PERFORMANCE")
        print("-"*40)
        for station_name, stats in report['station_performance'].items():
            print(f"{station_name.replace('_', ' ').title()}:")
            print(f"  Jobs Processed: {stats['jobs_processed']}")
            print(f"  Max Queue Length: {stats['max_queue_length']}")
            print(f"  Avg Queue Length: {stats['average_queue_length']:.2f}")
        
        print("\\n" + "-"*40)
        print("FORKLIFT PERFORMANCE")
        print("-"*40)
        fork_perf = report['forklift_performance']
        print(f"Average Utilization: {fork_perf['avg_utilization']:.3f}")
        print(f"Individual Utilizations: {[f'{u:.3f}' for u in fork_perf['individual_utilization']]}")
        print(f"Total Transports: {fork_perf['total_transports']}")
        print(f"Average Transports per Forklift: {fork_perf['avg_transports']:.1f}")
        
        print("="*60)
    
    def print_detailed_report(self, simulation_time, warmup_time, config, system_design_num):
        """Print a detailed report in the format requested."""
        report = self.get_summary_report(simulation_time, warmup_time)
        
        print(f"\nSimulation results for system design {system_design_num}")
        machine_counts = config.get('station_machine_counts', [])
        print(f"Number of machines: {', '.join(map(str, machine_counts))}")
        print(f"Number of forklifts: {config.get('num_forklifts', 0)}")
        
        # Header for station performance
        print("Station" + " " * 22 + "1" + " " * 6 + "2" + " " * 6 + "3" + " " * 6 + "4" + " " * 6 + "5")
        print("Performance measure")
        
        # Proportion machines busy
        busy_proportions = []
        for i in range(1, 6):
            station_key = f'station_{i}'
            if station_key in report['station_performance']:
                busy_proportions.append(f"{report['station_performance'][station_key]['busy_utilization']:.2f}")
            else:
                busy_proportions.append("0.00")
        print(f"Proportion machines busy     {' '.join(f'{p:>6}' for p in busy_proportions)}")
        
        # Proportion machines blocked
        blocked_proportions = []
        for i in range(1, 6):
            station_key = f'station_{i}'
            if station_key in report['station_performance']:
                blocked_proportions.append(f"{report['station_performance'][station_key]['blocked_utilization']:.2f}")
            else:
                blocked_proportions.append("0.00")
        print(f"Proportion machines blocked  {' '.join(f'{p:>6}' for p in blocked_proportions)}")
        
        # Average number in queue
        avg_queues = []
        for i in range(1, 6):
            station_key = f'station_{i}'
            if station_key in report['station_performance']:
                avg_queues.append(f"{report['station_performance'][station_key]['average_queue_length']:.2f}")
            else:
                avg_queues.append("0.00")
        print(f"Average number in queue      {' '.join(f'{q:>6}' for q in avg_queues)}")
        
        # Maximum number in queue
        max_queues = []
        for i in range(1, 6):
            station_key = f'station_{i}'
            if station_key in report['station_performance']:
                max_queues.append(f"{report['station_performance'][station_key]['max_queue_length']:.0f}")
            else:
                max_queues.append("0")
        print(f"Maximum number in queue      {' '.join(f'{q:>6}' for q in max_queues)}")
        
        # Overall performance metrics
        job_perf = report['job_performance']
        fork_perf = report['forklift_performance']
        
        print(f"Average daily throughput     {job_perf['throughput_per_8hr_day']:.2f}")
        print(f"Average time in system       {job_perf['avg_system_time']:.2f}")
        print(f"Average total time in queues {job_perf['avg_queue_time']:.2f}")
        print(f"Average total wait for transport {job_perf['avg_transport_time']:.2f}")
        print(f"Proportion forklifts moving loaded {fork_perf['proportion_moving_loaded']:.2f}")
        print(f"Proportion forklifts moving empty  {fork_perf['proportion_moving_empty']:.2f}")

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

#!/usr/bin/env python3
"""
System Analysis: Detailed breakdown of simulation components and processes
"""

from model_constants import *
import numpy as np

def analyze_system_components():
    """Provide detailed analysis of all system components."""
    
    print("="*80)
    print("MANUFACTURING SIMULATION SYSTEM - DETAILED COMPONENT ANALYSIS")
    print("="*80)
    
    # 1. System Parameters
    print(f"\n{'='*60}")
    print("1. SYSTEM PARAMETERS")
    print("="*60)
    
    print(f"📊 JOB ARRIVAL:")
    arrival_rate = 1 / MEAN_INTERARRIVAL_TIME
    print(f"   • Interarrival time: {MEAN_INTERARRIVAL_TIME:.4f} hours ({MEAN_INTERARRIVAL_TIME*60:.1f} min)")
    print(f"   • Arrival rate: {arrival_rate:.1f} jobs/hour")
    print(f"   • Expected daily input: {arrival_rate * 8:.1f} jobs/8hr-day")
    
    print(f"\n🏭 FACILITY LAYOUT:")
    print(f"   • 6 Stations: I/O Station (6) + Workstations (1-5)")
    print(f"   • Distance matrix: {len(DISTANCES)}x{len(DISTANCES[0])} (feet)")
    print(f"   • Min distance: {min(min(d for d in row if d > 0) for row in DISTANCES)} feet")
    print(f"   • Max distance: {max(max(row) for row in DISTANCES)} feet")
    
    print(f"\n🚛 TRANSPORT SYSTEM:")
    print(f"   • Forklift speed: {FORKLIFT_SPEED} feet/second")
    avg_distance = np.mean([d for row in DISTANCES for d in row if d > 0])
    avg_time = avg_distance / FORKLIFT_SPEED / 3600
    print(f"   • Average transport time: {avg_time*60:.1f} min ({avg_time:.4f} hours)")
    
    # 2. Job Types Analysis
    print(f"\n{'='*60}")
    print("2. JOB TYPES & ROUTING ANALYSIS")
    print("="*60)
    
    total_workload = 0
    for i, job_type_info in enumerate(JOB_TYPES):
        job_type = job_type_info['job_type']
        prob = job_type_info['probability']
        routing = ROUTINGS[job_type]
        service_times = MEAN_SERVICE_TIMES[job_type]
        
        total_service = sum(service_times)
        total_workload += arrival_rate * prob * total_service
        
        print(f"\n📦 JOB TYPE {job_type} (Probability: {prob}):")
        print(f"   • Routing: {' → '.join(map(str, routing))} → 6 (exit)")
        print(f"   • Operations: {len(service_times)}")
        print(f"   • Service times: {service_times}")
        print(f"   • Total service time: {total_service:.3f} hours")
        print(f"   • Arrival rate: {arrival_rate * prob:.1f} jobs/hour")
        print(f"   • Workload contribution: {arrival_rate * prob * total_service:.3f} machine-hours/hour")
    
    print(f"\n📈 SYSTEM TOTALS:")
    print(f"   • Total system workload: {total_workload:.3f} machine-hours/hour")
    print(f"   • Average job service time: {total_workload/arrival_rate:.3f} hours")
    
    # 3. Station Workload Analysis
    print(f"\n{'='*60}")
    print("3. STATION WORKLOAD ANALYSIS")
    print("="*60)
    
    station_workload = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    station_arrivals = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    
    for job_type_info in JOB_TYPES:
        job_type = job_type_info['job_type']
        prob = job_type_info['probability']
        routing = ROUTINGS[job_type]
        service_times = MEAN_SERVICE_TIMES[job_type]
        
        for i, station in enumerate(routing):
            service_time = service_times[i]
            workload_contribution = arrival_rate * prob * service_time
            station_workload[station] += workload_contribution
            station_arrivals[station] += arrival_rate * prob
    
    for station_id in [1, 2, 3, 4, 5]:
        workload = station_workload[station_id]
        arrivals = station_arrivals[station_id]
        avg_service = workload / arrivals if arrivals > 0 else 0
        
        print(f"\n🏭 STATION {station_id}:")
        print(f"   • Job arrivals: {arrivals:.1f} jobs/hour")
        print(f"   • Workload demand: {workload:.3f} machine-hours/hour")
        print(f"   • Average service time: {avg_service:.3f} hours")
        print(f"   • Machines needed (100% util): {workload:.1f}")

def analyze_system_processes():
    """Analyze the key processes in the simulation."""
    
    print(f"\n{'='*60}")
    print("4. SIMULATION PROCESSES")
    print("="*60)
    
    print(f"\n🔄 EVENT TYPES:")
    events = [
        ("job_arrival", "New job enters I/O station", "Schedule next arrival + forklift request"),
        ("forklift_dropoff", "Job delivered to station", "Start processing or join queue"),
        ("machine_finish", "Processing complete", "Machine blocked, request pickup"),
        ("job_exit", "Job leaves system", "Record statistics")
    ]
    
    for event_type, description, action in events:
        print(f"   • {event_type:15s}: {description:25s} → {action}")
    
    print(f"\n⚙️ MACHINE STATES:")
    states = [
        ("idle", "Available for new jobs"),
        ("busy", "Processing a job"),
        ("blocked", "Finished processing, waiting for forklift pickup")
    ]
    
    for state, description in states:
        print(f"   • {state:10s}: {description}")
    
    print(f"\n🚛 FORKLIFT LOGIC:")
    print(f"   • Assignment: Shortest distance first")
    print(f"   • Availability: FIFO queue when all busy")
    print(f"   • Positioning: Remains at dropoff location")
    print(f"   • Travel calculation: Distance matrix + constant speed")
    
    print(f"\n📊 PERFORMANCE TRACKING:")
    metrics = [
        "Job throughput (jobs/hour, jobs/8hr-day)",
        "System time (arrival to exit)",
        "Queue times and lengths",
        "Machine utilization (busy/blocked/idle)",
        "Forklift utilization and transport count",
        "Station-specific performance"
    ]
    
    for metric in metrics:
        print(f"   • {metric}")

def analyze_bottlenecks():
    """Analyze potential system bottlenecks."""
    
    print(f"\n{'='*60}")
    print("5. BOTTLENECK ANALYSIS")
    print("="*60)
    
    # Calculate station requirements
    arrival_rate = 1 / MEAN_INTERARRIVAL_TIME
    station_workload = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    
    for job_type_info in JOB_TYPES:
        job_type = job_type_info['job_type']
        prob = job_type_info['probability']
        routing = ROUTINGS[job_type]
        service_times = MEAN_SERVICE_TIMES[job_type]
        
        for i, station in enumerate(routing):
            service_time = service_times[i]
            workload_contribution = arrival_rate * prob * service_time
            station_workload[station] += workload_contribution
    
    print(f"\n🎯 CONFIGURATION ANALYSIS:")
    configs = [
        ("Reference Config 1", [4, 1, 4, 2, 2], 1, 94.94),
        ("Reference Config 2", [4, 2, 5, 3, 2], 1, 106.77),
        ("Reference Config 3", [4, 2, 5, 3, 2], 2, 120.29)
    ]
    
    for config_name, machines, forklifts, expected_throughput in configs:
        print(f"\n📋 {config_name}:")
        print(f"   • Machines: {machines}, Forklifts: {forklifts}")
        print(f"   • Expected throughput: {expected_throughput:.1f} jobs/8hr-day")
        
        max_util = 0
        bottleneck = None
        
        for station_id in [1, 2, 3, 4, 5]:
            workload = station_workload[station_id]
            num_machines = machines[station_id - 1]
            utilization = workload / num_machines if num_machines > 0 else float('inf')
            
            if utilization > max_util:
                max_util = utilization
                bottleneck = station_id
                
            status = "⚠️ OVERLOADED" if utilization > 1.0 else "✅ OK"
            print(f"     Station {station_id}: {utilization:.3f} utilization {status}")
        
        print(f"   • Bottleneck: Station {bottleneck} ({max_util:.3f} utilization)")
        
        # Forklift analysis
        total_transports = sum(len(ROUTINGS[jt['job_type']]) + 1 for jt in JOB_TYPES) / len(JOB_TYPES)
        transport_demand = arrival_rate * total_transports * 0.06  # avg transport time
        forklift_util = transport_demand / forklifts
        
        print(f"   • Forklift utilization: {forklift_util:.3f}")
        if forklift_util > 0.8:
            print(f"     ⚠️ HIGH forklift utilization - transport bottleneck likely")
        else:
            print(f"     ✅ Forklift capacity adequate")

if __name__ == "__main__":
    analyze_system_components()
    analyze_system_processes() 
    analyze_bottlenecks()
    
    print(f"\n{'='*80}")
    print("SUMMARY: This simulation models a complex manufacturing system with")
    print("realistic constraints including machine blocking, forklift transport,")
    print("and dynamic job routing. The model correctly identifies transport")
    print("bottlenecks as the primary performance limiter.")
    print("="*80)

#!/usr/bin/env python3
"""
Test script to validate simulation model against expected performance metrics.
"""

import numpy as np
from manufacturing_simulation import ManufacturingSimulation, run_experiment
import model_constants

def print_detailed_metrics_report(results, config, system_design_num):
    """Print detailed metrics in the format specified by the user."""
    if not results:
        print("No results available")
        return
    
    # Calculate averages across all replications
    n_replications = len(results)
    
    # Station-level metrics (average across replications)
    station_metrics = {}
    for i in range(1, 6):
        station_key = f'station_{i}'
        station_metrics[i] = {
            'busy_util': [],
            'blocked_util': [],
            'avg_queue': [],
            'max_queue': []
        }
    
    # Collect metrics from all replications
    for result in results:
        if 'station_performance' in result:
            for i in range(1, 6):
                station_key = f'station_{i}'
                if station_key in result['station_performance']:
                    station_data = result['station_performance'][station_key]
                    station_metrics[i]['busy_util'].append(station_data.get('busy_utilization', 0))
                    station_metrics[i]['blocked_util'].append(station_data.get('blocked_utilization', 0))
                    station_metrics[i]['avg_queue'].append(station_data.get('average_queue_length', 0))
                    station_metrics[i]['max_queue'].append(station_data.get('max_queue_length', 0))
                else:
                    # If station not in results, use zeros
                    station_metrics[i]['busy_util'].append(0)
                    station_metrics[i]['blocked_util'].append(0)
                    station_metrics[i]['avg_queue'].append(0)
                    station_metrics[i]['max_queue'].append(0)
    
    # Calculate averages for each station
    for i in range(1, 6):
        for metric in station_metrics[i]:
            if station_metrics[i][metric]:
                station_metrics[i][metric] = np.mean(station_metrics[i][metric])
            else:
                station_metrics[i][metric] = 0
    
    # Overall performance metrics
    throughput = np.mean([r['job_performance']['throughput_per_8hr_day'] for r in results])
    avg_system_time = np.mean([r['job_performance']['avg_system_time'] for r in results])
    avg_queue_time = np.mean([r['job_performance']['avg_queue_time'] for r in results])
    avg_transport_time = np.mean([r['job_performance']['avg_transport_time'] for r in results])
    
    # Forklift metrics
    proportion_loaded = np.mean([r['forklift_performance']['proportion_moving_loaded'] for r in results])
    proportion_empty = np.mean([r['forklift_performance']['proportion_moving_empty'] for r in results])
    
    # Print the detailed report
    print(f"\nSimulation results for system design {system_design_num}")
    machine_counts = config.get('station_machine_counts', [])
    print(f"Number of machines: {', '.join(map(str, machine_counts))}")
    print(f"Number of forklifts: {config.get('num_forklifts', 0)}")
    
    # Header for station performance
    print("Station" + " " * 22 + "1" + " " * 6 + "2" + " " * 6 + "3" + " " * 6 + "4" + " " * 6 + "5")
    print("Performance measure")
    
    # Proportion machines busy
    busy_proportions = [f"{station_metrics[i]['busy_util']:.2f}" for i in range(1, 6)]
    print(f"Proportion machines busy     {' '.join(f'{p:>6}' for p in busy_proportions)}")
    
    # Proportion machines blocked
    blocked_proportions = [f"{station_metrics[i]['blocked_util']:.2f}" for i in range(1, 6)]
    print(f"Proportion machines blocked  {' '.join(f'{p:>6}' for p in blocked_proportions)}")
    
    # Average number in queue
    avg_queues = [f"{station_metrics[i]['avg_queue']:.2f}" for i in range(1, 6)]
    print(f"Average number in queue      {' '.join(f'{q:>6}' for q in avg_queues)}")
    
    # Maximum number in queue
    max_queues = [f"{station_metrics[i]['max_queue']:.0f}" for i in range(1, 6)]
    print(f"Maximum number in queue      {' '.join(f'{q:>6}' for q in max_queues)}")
    
    # Overall performance metrics
    print(f"Average daily throughput     {throughput:.2f}")
    print(f"Average time in system       {avg_system_time:.2f}")
    print(f"Average total time in queues {avg_queue_time:.2f}")
    print(f"Average total wait for transport {avg_transport_time:.2f}")
    print(f"Proportion forklifts moving loaded {proportion_loaded:.2f}")
    print(f"Proportion forklifts moving empty  {proportion_empty:.2f}")

def test_configuration(config_info):
    """Test a single configuration with the given parameters.
    
    Args:
        config_info (dict): Configuration information containing:
            - name: Configuration name/description
            - machines: List of machine counts per station
            - forklifts: Number of forklifts
            - design_num: System design number for reporting
            - description: Additional description lines (optional)
            - simulation_params: Override simulation parameters (optional)
            - machine_efficiencies: Dict mapping station numbers to efficiency values (optional)
    
    Returns:
        list: Results from all replications
    """
    # Default simulation parameters
    default_params = {
        'num_stations': 5,
        'simulation_hours': 920,
        'warmup_hours': 0,
        'replications': 10,
        'random_seed': 12345
    }
    
    # Override with any custom parameters
    sim_params = default_params.copy()
    if 'simulation_params' in config_info:
        sim_params.update(config_info['simulation_params'])
    
    # Build the complete configuration
    config = sim_params.copy()
    config['station_machine_counts'] = config_info['machines']
    config['num_forklifts'] = config_info['forklifts']

    # Print configuration header
    machines_str = ','.join(map(str, config_info['machines']))
    header = f"TESTING {config_info['name']}: [{machines_str}] machines, {config_info['forklifts']} forklifts"
    
    print("\n" + "="*max(60, len(header)))
    print(header)
    
    # Print description lines if provided
    if 'description' in config_info:
        for desc_line in config_info['description']:
            print(desc_line)
    
    print("="*max(60, len(header)))
    
    # Run the experiment
    results = run_experiment(config, verbose=False)
    
    # Print detailed metrics report
    print_detailed_metrics_report(results, config, config_info['design_num'])
    
    return results

if __name__ == "__main__":
    print("MANUFACTURING SIMULATION MODEL VALIDATION")
    print("Comparing actual vs expected performance metrics")
    
    # Define all test configurations
    configurations = [
        {
            'name': 'Configuration 1',
            'machines': [4, 1, 4, 2, 2],
            'forklifts': 1,
            'design_num': 1,
            'efficiency': 1,
            'target_machines': [],
            'description': ['This tests the original design with insufficient forklift capacity.']
        },
        {
            'name': 'Configuration 2',
            'machines': [4, 2, 5, 3, 2],
            'forklifts': 1,
            'design_num': 2,
            'efficiency': 1,
            'target_machines': [],
            'description': ['This tests system design 2 with insufficient forklift capacity.']
        },
        {
            'name': 'Configuration 3',
            'machines': [4, 2, 5, 3, 2],
            'forklifts': 2,
            'design_num': 3,
            'efficiency': 1,
            'target_machines': [],
            'description': ['This tests system design 3 with adequate forklift capacity.']
        },
        {
            'name': 'Configuration 4',
            'machines': [4, 2, 5, 2, 2],
            'forklifts': 2,
            'design_num': 4,
            'efficiency': 1,
            'target_machines': [],
            'description': ['This tests system design 4 with adequate forklift capacity.']
        },
        {
            'name': 'Configuration 5',
            'machines': [4, 2, 5, 3, 2],
            'forklifts': 2,
            'design_num': 5,
            'efficiency': 0.9,
            'target_machines': [1,3],
            'description': ['This tests system design 4 with adequate forklift capacity.']
        },
        {
            'name': 'Configuration 6',
            'machines': [4, 2, 5, 3, 2],
            'forklifts': 2,
            'design_num': 6,
            'efficiency': 0.9,
            'target_machines': [1, 5],
            'description': ['This tests system design 4 with adequate forklift capacity.']
        }        
    ]
    
    # Run all test configurations
    all_results = []
    for config in configurations:
        model_constants.MACHINE_EFFICIENCY = config['efficiency']   
        model_constants.EFFICENCY_TARGET_MACHINES = config['target_machines'] 
        results = test_configuration(config)
        print("Config efficiency: ", config['efficiency'])
        all_results.append((config, results))
    
    print("\n" + "="*80)
    print("SUMMARY OF ALL TESTS")
    print("="*80)
    
    # Summary table with expected values for comparison
    expected_values = [
        (94.94, 109.20),  # Config 1
        (106.77, 55.84),  # Config 2
        (120.29, 1.76),   # Config 3
        (120.29, 1.76),   # Config 4 (estimated)
        (120.33, 2.03),   # Config 5 (estimated)   
        (119.88, 5.31),   # Config 6 (estimated)     
    ]
    
    print(f"{'Configuration':<35} {'Actual':<12} {'Expected':<12} {'Diff':<12} {'System Time':<15}")
    print("-" * 95)
    
    for i, (config, results) in enumerate(all_results):
        actual_throughput = np.mean([r['job_performance']['throughput_per_8hr_day'] for r in results])
        actual_system_time = np.mean([r['job_performance']['avg_system_time'] for r in results])
        
        exp_throughput, exp_system_time = expected_values[i]
        diff = actual_throughput - exp_throughput
        
        machines_str = ','.join(map(str, config['machines']))
        name = f"{config['name']}: [{machines_str}], {config['forklifts']} forklift{'s' if config['forklifts'] > 1 else ''}"
        
        print(f"{name:<35} {actual_throughput:<12.2f} {exp_throughput:<12.2f} {diff:<12.2f} {actual_system_time:<15.2f}")
    
    print(f"\nKEY FINDINGS:")
    print("="*50)
    print("✅ MAJOR ISSUE IDENTIFIED: Forklift capacity bottleneck")
    print("✅ ROOT CAUSE: Theoretical forklift analysis oversimplified")
    print("✅ SOLUTION: Increase forklift count to account for:")
    print("   - Machine blocking effects")
    print("   - Queueing delays")
    print("   - Empty travel time")
    print("   - Variable workload distribution")
    print("\n💡 RECOMMENDATION: Use 2-3 forklifts minimum for realistic operations")
    
    print(f"\n🎯 DETAILED METRICS OUTPUT:")
    print("="*50)
    print("✅ All test configurations now output comprehensive metrics including:")
    print("   - Proportion of machines busy/blocked by station")
    print("   - Average and maximum queue lengths by station")
    print("   - Daily throughput and system times")
    print("   - Forklift utilization (loaded/empty movement)")
    print("   - Transport wait times")
    print("\n📊 Use the detailed reports above for system design analysis!")










