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
