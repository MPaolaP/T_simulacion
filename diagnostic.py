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
