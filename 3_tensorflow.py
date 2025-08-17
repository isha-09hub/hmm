#3. TensorFlow Computation Graph with Eager Execution
import tensorflow as tf

a = tf.constant([5.0, 3.0])
b = tf.constant([2.0, 7.0])
c = a + b
print("Eager Execution Output:", c.numpy())

@tf.function
def multiply_tensors(x, y):
    return x * y

result = multiply_tensors(a, b)
print("Graph Mode Output:", result.numpy())

