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
    def __init__(self, presence_initializer=HeNormal(), xy_initializer=HeNormal(), dimensions_initializer=HeNormal(), class_initializer=HeNormal(), classes=2, anchors=3):
        self.presence_initializer = presence_initializer
        self.xy_initializer = xy_initializer
        self.dimensions_initializer = dimensions_initializer
        self.class_initializer = class_initializer

        self.classes = classes
        self.anchors = anchors

    def __call__(self, fan_in, fan_out, shape):

        presence_shape = list(shape)
        presence_shape = list(presence_shape[:-1] + [self.anchors, 1])

        dimensions_shape = list(shape)
        dimensions_shape = dimensions_shape[:-1] + [self.anchors, 2]

        classes_shape = list(shape)
        classes_shape = classes_shape[:-1] + [self.anchors, self.classes]

        xy_data = tf.cast(self.xy_initializer(fan_in, fan_out, dimensions_shape), tf.float64)
        dimensions_data = tf.cast(self.dimensions_initializer(fan_in, fan_out, dimensions_shape), tf.float64)
        presence_data = tf.cast(self.presence_initializer(fan_in, fan_out, presence_shape), tf.float64) # Every 5th output will correspond with a presence score
        class_data = tf.cast(self.class_initializer(fan_in, fan_out, classes_shape), tf.float64)

        data = tf.concat((presence_data, xy_data, dimensions_data, class_data), axis=-1)

        return tf.reshape(data, shape)