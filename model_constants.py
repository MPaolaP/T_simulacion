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
