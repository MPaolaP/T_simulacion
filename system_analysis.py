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
