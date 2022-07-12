class LearningRateSchedule:
    def get_learning_rate(self, epoch):
        pass


class ConstantLearningRateSchedule(LearningRateSchedule):
    def __init__(self, value):
        self.value = value

    def get_learning_rate(self, epoch):
        return self.value


class StepLearningRateSchedule(LearningRateSchedule):
    def __init__(self, initial, interval, factor):
        self.initial = initial
        self.interval = interval
        self.factor = factor

    def get_learning_rate(self, iter_num):

        return self.initial * (self.factor ** (iter_num // self.interval))


class WarmupLearningRateSchedule(LearningRateSchedule):
    def __init__(self, initial, warmed_up, length):
        self.initial = initial
        self.warmed_up = warmed_up
        self.length = length

    def get_learning_rate(self, epoch):
        if epoch > self.length:
            return self.warmed_up
        return self.initial + (self.warmed_up - self.initial) * epoch / self.length


def get_learning_rate_schedules(type, **kwargs):

    if type == 'Step':
        assert 'Initial' in kwargs, 'Missing keyword argument "Initial"'
        assert 'Interval' in kwargs, 'Missing keyword argument "Interval"'
        assert 'Factor' in kwargs, 'Missing keyword argument "Factor"'
        return StepLearningRateSchedule(
                    kwargs["Initial"],
                    kwargs["Interval"],
                    kwargs["Factor"],
                )
    elif type == 'Warmup':
        assert 'Initial' in kwargs, 'Missing keyword argument "Initial"'
        assert 'Final' in kwargs, 'Missing keyword argument "Final"'
        assert 'Length' in kwargs, 'Missing keyword argument "Length"'
        return WarmupLearningRateSchedule(
                    kwargs["Initial"],
                    kwargs["Final"],
                    kwargs["Length"],
                )
    elif type == 'Constant':
        assert 'Value' in kwargs, 'Missing keyword argument "Value"'
        return ConstantLearningRateSchedule(kwargs["Value"])
    else:
        raise ValueError(
            'Unknown learning rate of type "{}"! '
            'Schedule ype must be "Step", "Warmup" or "Constant". '.format(type))
