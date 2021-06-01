import numpy as np


# Schedulers for beta
class Scheduler(object):
    def __call__(self, **kwargs):
        raise NotImplementedError()


class LinearScheduler(Scheduler):
    def __init__(
        self, start_value, end_value, n_iterations, start_iteration=0
    ):
        self.start_value = start_value
        self.end_value = end_value
        self.n_iterations = n_iterations
        self.start_iteration = start_iteration
        self.m = (end_value - start_value) / n_iterations

    def __call__(self, iteration):
        if iteration > self.start_iteration + self.n_iterations:
            return self.end_value
        elif iteration <= self.start_iteration:
            return self.start_value
        else:
            return (
                (iteration - self.start_iteration) * self.m + self.start_value
            )


class ExponentialScheduler(LinearScheduler):
    def __init__(
        self, start_value, end_value, n_iterations, start_iteration=0,
    ):
        super(ExponentialScheduler, self).__init__(
            start_value=np.log10(start_value),
            end_value=np.log10(end_value),
            n_iterations=n_iterations,
            start_iteration=start_iteration
        )

    def __call__(self, iteration):
        linear_value = super(ExponentialScheduler, self).__call__(iteration)
        return np.power(10., linear_value)


class SmoothScheduler(object):
    def __init__(
        self, start_value, end_value, n_iterations,
        start_iteration=0, mid_point=0.25, beta=2.,
    ):
        self.start_value = start_value
        self.end_value = end_value
        self.n_iterations = n_iterations
        self.start_iteration = start_iteration
        self.mid_point = mid_point
        self.beta = beta

    def linear_schedule(
        self, step, init_step, total_step, init_value, final_value
    ):
        if step < init_step:
            return init_value
        rate = float(step - init_step) / \
            float(total_step + self.start_iteration - init_step)
        linear_value = rate * (final_value - init_value) + init_value
        return np.clip(linear_value, init_value, final_value)

    def __call__(self, iteration):
        mid_step = self.n_iterations * self.mid_point
        t = mid_step ** (self.beta - 1.)
        alpha = (self.end_value - self.start_value) / (
            (self.n_iterations - mid_step) * self.beta * t + mid_step * t
        )
        mid_value = alpha * mid_step ** self.beta + self.start_value
        is_ramp = float(iteration > self.start_iteration)
        is_linear = float(iteration >= mid_step + self.start_iteration)
        return (is_ramp * (
            (1. - is_linear) * (
                self.start_value + alpha * float(
                    iteration - self.start_iteration
                ) ** self.beta) +
            is_linear * self.linear_schedule(
                iteration, mid_step + self.start_iteration,
                self.n_iterations, mid_value, self.end_value
            )) + (1. - is_ramp) * self.start_value)
