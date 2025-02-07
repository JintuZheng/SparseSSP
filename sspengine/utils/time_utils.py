import time
from datetime import datetime, timedelta

class TrainingETA(object):
    def __init__(self, total_steps):
        self.total_steps = total_steps
        self.start_time = time.time()
        self.current_step = 0

    def step(self):
        self.current_step += 1

    def get_eta(self):
        if self.current_step == 0:
            return "No steps completed yet, ETA unavailable."
        elapsed_time = time.time() - self.start_time
        steps_left = self.total_steps - self.current_step
        average_time_per_step = elapsed_time / self.current_step
        remaining_time = average_time_per_step * steps_left
        elapsed_str = str(timedelta(seconds=int(elapsed_time)))
        remaining_str = str(timedelta(seconds=int(remaining_time)))
        return "Avg sec.: {:.1f}, Elapsed: {}, ETA: {}".format(average_time_per_step, elapsed_str, remaining_str)