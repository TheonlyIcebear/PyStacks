import cupy as cp

class HeNormal:
    def __init__(self):
        pass

    def forward(self, fan_in, fan_out, shape):
        variance = cp.sqrt(2 / (fan_in))
        return cp.random.normal(0, variance, shape)

class LecunNormal:
    def __init__(self):
        pass

    def forward(self, fan_in, fan_out, shape):
        variance = cp.sqrt(1 / (fan_in))
        return cp.random.normal(0, variance, shape)
        
class XavierUniform:
    def __init__(self):
        pass

    def forward(self, fan_in, fan_out, shape):
        variance = cp.sqrt(6 / (fan_in + fan_out))
        return cp.random.uniform(-variance, variance, shape)

class Uniform:
    def __init__(self, start=-0.05, stop=0.05):
        self.start = start
        self.stop = stop

    def forward(self, fan_in, fan_out, shape):
        return cp.random.uniform(self.start, self.stop, shape)

class Normal:
    def __init__(self, mean=0, std=1):
        self.mean = mean
        self.std = std

    def forward(self, fan_in, fan_out, shape):
        return cp.random.normal(self.mean, self.std, shape)

class Fill:
    def __init__(self, value=0):
        self.value = value

    def forward(self, fan_in, fan_out, shape):
        return cp.full(shape, self.value)

class YoloSplit:
    def __init__(self, presence_initializer=HeNormal(), dimensions_initializer=HeNormal()):
        self.presence_initializer = presence_initializer
        self.dimensions_initializer = dimensions_initializer

    def forward(self, fan_in, fan_out, shape):
        new_shape = list(shape)
        new_shape[-1] //= 5

        data = self.dimensions_initializer.forward(fan_in, fan_out, shape)
        data[..., ::5] = self.presence_initializer.forward(fan_in, fan_out, new_shape) # Every 5th output will correspond with a presence score

        return data