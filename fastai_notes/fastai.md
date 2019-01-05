# Main components of the fastai library

## Training

A `Learner` binds together a model, a dataset, an optimizer and a loss function. The `Learner` class is defined in `basic_train`. The training loop is operated by the `fit` method, and most customizations happen via the `Callback` objects.

The `Callback` class is defined in the `callback` module. `Callback` objects communicate with the training loop via the `CallbackHandler`, which maintains a state dictionary that is passed to the `Callback` during the training loop.

`callbacks` is actually a folder where each `Callback` is implemented in a separate module. On the "bimbo" it can be found in the folder
`~/miniconda3/envs/fastai_v1/lib/python3.6/site-packages/fastai`.

The `train` module provides a number of extension methods that are added to `Learner`. These methods are automatically added to all `Learner` objects created after imposing this module.

## Walk-through of key functionality

The main steps in using `fastai` are, in a nutshell:

1. Create a `DataBunch` object.
2. Choose a model.
3. Create a `Learner` object using the data and the model.
4. Fit the model.

To measure the progress of our model we can use one of various metrics. These can be passed to the constructor or set at a later stage, for example with:

```py
# Note that we are passing a list containing the `accuracy` object.
learn.metrics = [accuracy]
```

If the desired metric is not available, it can be easily added.

### Callbacks

Callbacks add functionality to the training loop. One example is the `OneCycleScheduler` callback. The `Recorder` callback is automatically added to a `Learner`, and it can be used to visualize the loss or the learning rate. `learn.recorder.plot()` creates a scatter-plot of the loss vs the learning rate. `learn.recorder.plot_losses()` plots the training and validation losses. `learn.plot_lr()` plots the learning rate and, optionally, if `show_moms=True` the momentum. `learn.recorder.plot_metrics()` plots the metrics collected during training. Since metrics are collected at the end of every epoch, you must train at least two epochs to visualize anything.

### Extending `Learner` with `train`

`train` contains a number of callbacks that can be more easily used with `Learner` objects. Above, we have created a `OneCycleScheduler` and passed it to the `Learner`. We can, however, just use the method `learn.fit_one_cycle()`. Similarly, one can use the `learn.find_lr` to find a good learning rate.

## `basic_train`

The `basic_train` module defines the `Learner` class, with the associated methods to execute the training loop, get the predictions, inspect them, summarize the results, run TTA, perform discriminative layer training etc.

