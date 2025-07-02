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
