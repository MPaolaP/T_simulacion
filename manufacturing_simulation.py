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
