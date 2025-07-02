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