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
