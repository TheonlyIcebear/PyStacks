import tensorflow.compat.v1 as tf

class HeNormal:
    def __init__(self):
        pass

    def __call__(self, fan_in, fan_out, shape):
        variance = tf.sqrt(2 / (fan_in))
        return tf.random.normal(shape, 0, variance)

class LecunNormal:
    def __init__(self):
        pass

    def __call__(self, fan_in, fan_out, shape):
        variance = tf.sqrt(1 / (fan_in))
        return tf.random.normal(shape, 0, variance)
        
class XavierUniform:
    def __init__(self):
        pass

    def __call__(self, fan_in, fan_out, shape):
        variance = tf.sqrt(6 / (fan_in + fan_out))
        return tf.random.uniform(shape, -variance, variance)

class Uniform:
    def __init__(self, start=-0.05, stop=0.05):
        self.start = start
        self.stop = stop

    def __call__(self, fan_in, fan_out, shape):
        return tf.random.uniform(shape, self.start, self.stop)

class Normal:
    def __init__(self, mean=0, std=1):
        self.mean = mean
        self.std = std

    def __call__(self, fan_in, fan_out, shape):
        return tf.random.normal(shape, self.mean, self.std)

class Fill:
    def __init__(self, value=0):
        self.value = value

    def __call__(self, fan_in, fan_out, shape):
        return tf.fill(shape, self.value)

class YoloSplit:
    def __init__(self, presence_initializer=HeNormal(), xy_initializer=HeNormal(), dimensions_initializer=HeNormal()):
        self.presence_initializer = presence_initializer
        self.xy_initializer = xy_initializer
        self.dimensions_initializer = dimensions_initializer

    def __call__(self, fan_in, fan_out, shape):
        anchors = shape[-1] // 5

        presence_shape = list(shape)
        presence_shape[-1] //= 5
        presence_shape = presence_shape[:-1] + [anchors, 1]

        dimensions_shape = list(shape)
        dimensions_shape[-1] = int((dimensions_shape[-1] - anchors) / 2)
        dimensions_shape = dimensions_shape[:-1] + [anchors, dimensions_shape[-1] // anchors]

        xy_data = tf.cast(self.xy_initializer(fan_in, fan_out, dimensions_shape), tf.float64)
        dimensions_data = tf.cast(self.dimensions_initializer(fan_in, fan_out, dimensions_shape), tf.float64)
        presence_data = tf.cast(self.presence_initializer(fan_in, fan_out, presence_shape), tf.float64) # Every 5th output will correspond with a presence score

        data = tf.concat((presence_data, xy_data, dimensions_data), axis=-1)

        return tf.reshape(data, shape)