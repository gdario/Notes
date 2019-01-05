# Introduction to TensorFlow variables

Variables represent tensors that whose value can be modified by running ops. They exist outside the context of a single `session.run` call. Modifications introduced by ops are visible across multiple `tf.session`s, so multiple workers can see the same values for a variable

## Creating a variable

Use `tf.get_variable`. It accepts the following arguments:

```py
tf.get_variable(name, shape=None, dtype=None, initializer=None,
	regularizer=None, trainable=True, 
	collections=None, caching_device=None, 
	partitioner=None, validate_shape=True, 
	use_resource=None, custom_getter=None)
```

You can define a variable just with `name` and `shape`. By default `dtype` is `tf.float32` and the initializer is `tf.glorot_uniform_intializer`. All these can be changed. If you want a variable to contain the value of an existing tensor, you can pass it to the initializer:

```py
my_tensor = tf.constant([23, 42], dtype=tf.int32, name='my_tensor')
w = tf.get_variable('w', dtype=tf.int32, initializer=my_tensor)
```

## Collections

Don't confuse collections with variable scopes. They are completely different concepts. A collection is just a way to access related variables. For example, the `tf.Optimizer` subclasses default to optimizing the variables collected under `tf.GraphKeys.TRAINABLE_VARIABLES`. In general, there are some pre-existing collections, as `GLOBAL_VARIABLES`, `LOCAL_VARIABLES`, and many others. The `GLOBAL_VARIABLES` collection contains the `MODEL_VARIABLES` collection that, in turn, contains the `TRAINABLE_VARIABLES` collection. One important collection for regularization purposes is the `REGULARIZATION_LOSSES`. The standard names for graph collections are stored in `tf.GraphKeys`, and by typing `help(tf.GraphKeys)` you can have an overview of the many collections available. 

If we define a variable but we don't want it to be trained, we can add it to the `LOCAL_VARIABLES` collections, or specify `trainable=False` when we create it.

We can create our own collections. For example, both commands below will add the variables `v` and `w` to the collection `my_collection`.

```py
v = tf.get_variable('v', [2, 2], collections=['my_collection])
w = tf.get_variable('w', [3, 3])
tf.add_to_collection('my_collection', w)
```

To see the content of a collection, we can use `tf.get_collection`, which is just a wrapper around `Graph.get_collection()` using the default graph. If you pass the optional `scope` argument you will get only the elements of the collection whose names match `scope`. Items without a name are not returned.

```py
tf.get_variable('v2', (), collection=['my_collection'])
tf.get_collection('my_collection', scope='v')
```

Note that `tf.get_variable` gets a variable or creates one, if it does not exist. This is a safety mechanism, so that you don't risk to overwrite existing variables unless you specify `reuse=True`. Note that, unlike `tf.get_variable`, the `tf.Variable` constructor requires an initial value.

## Device placement

You can specify on which device a variable should be placed with a device scope.

```py
with tf.device('/gpu:1'):
	v = tf.get_variable('v', [1])
```

If you are working with multiple servers, with a parameter server and one or more workers, you should make sure that variables are placed in the former and not in the latter, as this may impact performance. Specify `tf.train.replica_device_setter` to automatically have variables placed on the parameter servers.

```py
with tf.device(tf.train.replica_device_setter(cluster=cluster_spec)):
	v = tf.get_variable('v', shape=[10, 10])
```

## Variable initialization

Initialization is explicit to avoid rerunning expensive operations when reloading a model from a checkpoint. To initialize all variables use `tf.global_variables_initializer()`. This op initializes all variables in `tf.GraphKeys.GLOBAL_VARIABLES`. You can initialize variables *by hand* as in:

```py
session.run(my_var.initializer)
```

You can check which variables are not initialized with:

```py
print(session.run(tf.report_uninitialized_variables()))
```

### Order of initialization

The global initializer does not specify the order of initialization. If the initialization of a variable depends on another variable, use `variable.initialized_value()` as in:

```py
v = tf.get_variable('v', shape=(), initializer=tf.zeros_initializer())
w = tf.get_variable('w', initializer=v.initialized_value() + 1)
```

This forces the initialization of `w` to "wait" for the initialization of `v`.

## Using variables

You can use variables as normal tensors. You can assign values by `assign`. You can update a variable by adding the value of a tensor with `tf.assign_add`. In the code below, pay attention to the fact that the `assignment` node is performing the assignment operation, which must be also run.

```py
v = tf.get_variable('v', shape=(), initializer=tf.zero_initializer())
assignment = v.assign_add(1)
tf.global_variables_initializer().run()
assignment.run()
```

You can re-read the value of a variable after something has happened with `read_value()` as shown below:

```py
v = tf.get_variable("v", shape=(), initializer=tf.zeros_initializer())
assignment = v.assign_add(1)
with tf.control_dependencies([assignment]):
    w = v.read_value()  # w is guaranteed to reflect v's value after the
                        # assign_add operation.
```

## Sharing variables

There are two ways of sharing variables: explicit and implicit.

### Implicit approach - variable scopes

Most `tf.layers` as well as all `tf.metrics` use this approach. Following the example on the Programmer's Guide, let's create a function that returns a convolutional layer.

```py
def conv_relu(input, kernel_shape, bias_shape):
    # Create variable named "weights".
    weights = tf.get_variable("weights", kernel_shape,
        initializer=tf.random_normal_initializer())
    # Create variable named "biases".
    biases = tf.get_variable("biases", bias_shape,
        initializer=tf.constant_initializer(0.0))
    conv = tf.nn.conv2d(input, weights,
        strides=[1, 1, 1, 1], padding='SAME')
    return tf.nn.relu(conv + biases)
```

The function creates the `weights` and `biases` variables. Let's an input tensor, apply a convolution, and re-apply the convolution to what we obtain. This will not work because TF does not know whether we want to create new variables or reuse them.

```py
input1 = tf.random_normal([1,10,10,32])
x = conv_relu(input1, kernel_shape=[5, 5, 1, 32], bias_shape=[32])
x = conv_relu(x, kernel_shape=[5, 5, 32, 32], bias_shape = [32])  # This fails.
```

If we want new variables, we need to specify different variable scopes:

```py
def my_image_filter(input_images):
    with tf.variable_scope("conv1"):
        # Variables created here will be named "conv1/weights", "conv1/biases".
        relu1 = conv_relu(input_images, [5, 5, 1, 32], [32])
    with tf.variable_scope("conv2"):
        # Variables created here will be named "conv2/weights", "conv2/biases".
        return conv_relu(relu1, [5, 5, 32, 32], [32])
```

If we want to reuse the weights, we must be explicit:

```py
with tf.variable_scope("model"):
  output1 = my_image_filter(input1)
with tf.variable_scope("model", reuse=True):
  output2 = my_image_filter(input2)
```

Alternatively, we can use `scope.reuse_variables()` as shown below:

```py
with tf.variable_scope("model") as scope:
  output1 = my_image_filter(input1)
  scope.reuse_variables() # This forces variable reuse
  output2 = my_image_filter(input2)
```

You can initialize a variable scope based on an existing one.

```py
with tf.variable_scope("model") as scope:
  output1 = my_image_filter(input1)
with tf.variable_scope(scope, reuse=True):
  output2 = my_image_filter(input2)
```
