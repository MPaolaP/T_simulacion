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
