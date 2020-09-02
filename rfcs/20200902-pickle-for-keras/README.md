# Title of RFC

| Status        | (Proposed / Accepted / Implemented / Obsolete)       |
:-------------- |:---------------------------------------------------- |
| **RFC #**     | [286](https://github.com/tensorflow/community/pull/286) |
| **Author(s)** | Adrian Garcia Badaracco ({firstname}@{firstname}gb.com), Scott Sievert (tf-rfc@stsievert.com)  |
| **Sponsor**   | Mihai Maruseac (mihaimaruseac@google.com)                 |
| **Updated**   | 2020-09-02                                           |

## Objective

Implement support for Python's Pickle protocol within Keras.

## Motivation

> *Why this is a valuable problem to solve? What background information is needed
to show how this design addresses the problem?*
The specific motivation
for this RFC comes from supporting Keras models in Dask-ML's and Ray's hyperparameter
optimization. More generaly, support for serialization with the Pickle protocol will enable:

* Using Keras with distributed systems (e.g, Python's `multiprocessing`, Dask, Ray or IPython parallel).
* Copying Keras models with Python's built-in `copy.deepcopy`.

This request is *not* advocating for use of Pickle while saving or sharing
Keras models. We believe the efficient, secure and stable methods in TF should be
used for that. We are proposing to add a Pickle implementation that uses the same efficient method.
This will enable wider usage in the Python ecosystem.

> *Which users are affected by the problem? Why is it a problem? What data supports
this? What related work exists?*
Users trying to use distributed systems (e.g, Ray or Dask) with Keras are 
affected. In our experience, this is common in hyperparameter optimization.

Related work is in [SciKeras], which brings a Scikit-Learn API
to Keras. Scikit-Learn estimators must be able to be pickled ([source][skp]). As such,
SciKeras has an implementation of `__reduce_ex__`, which is also in
[tensorflow#39609].

[tensorflow#39609]:https://github.com/tensorflow/tensorflow/pull/39609
[SciKeras]:https://github.com/adriangb/scikeras
[skp]:https://github.com/scikit-learn/scikit-learn/blob/0fb307bf39bbdacd6ed713c00724f8f871d60370/sklearn/utils/estimator_checks.py#L1523-L1524

Here are several examples
of Keras users running into issues because pickle is not supported.

GH issues where `Model.save` would not work:

* [TF#34697](https://github.com/tensorflow/tensorflow/issues/34697)
* [TF#33204](https://github.com/tensorflow/tensorflow/issues/33204)
* TODO: issues from Dask

StackOverflow questions where `Model.save` would not work:

* [SO#59872509](https://stackoverflow.com/questions/59872509/how-to-export-a-model-created-from-kerasclassifier-and-gridsearchcv-using-joblib)
* [SO#37984304](https://stackoverflow.com/questions/37984304/how-to-save-a-scikit-learn-pipline-with-keras-regressor-inside-to-disk)
* [SO#51110834](https://stackoverflow.com/questions/51110834/cannot-pickle-dill-a-keras-object)
* [SO#54070845](https://stackoverflow.com/questions/54070845/how-to-pickle-keras-custom-layer)
* [SO#40396042](https://stackoverflow.com/questions/40396042/how-to-save-scikit-learn-keras-model-into-a-persistence-file-pickle-hd5-json-ya)
* [SO#48295661](https://stackoverflow.com/questions/48295661/how-to-pickle-keras-model)
  
Examples that could be resolved using `Model.save` (but the user tried pickle first):

* [SO #51878627](https://stackoverflow.com/questions/51878627/pickle-keras-ann)

Here is an post describing how PyTorch ran into similar issues and how they resolved them: [link](https://matthewrocklin.com/blog/work/2018/07/23/protocols-pickle)

## User Benefit

> How will users (or other contributors) benefit from this work? What would be the headline in the release notes or blog post?

One blog post headline: "Keras models can be used in advanced hyperparameter optimization with Ray Tune or Dask-ML."

Users will notice:

* Improved compatibility with libraries like Scikit-Learn and Dask.
* Smaller learning curve for new Keras/TF users since they will be able to
use entry points they already know and simultaneously access TFs high performance
internal methods.

## Design Proposal

The pickle protocol supports two distinct functions:

1. In-memory copying of live objects: via Python's `copy` module. This falls back to (2) below.
2. Serialization to arbitrary IO (memory or disk): via Python's `pickle` module.

This proposal seeks to take the conservative approach at least initially and only
implement (2) above since (1) can always fall back to (2) and using only (2) alleviates
any concerns around references to freed memory in the C++ portions of TF and other such bugs.

This said, for situations where the user is making an in-memory copy of an object and it might
even be okay to keep around references to non-Python objects, a separate approach that optimizes
(1) would be warranted. This RFC does not seek to address this problem. Hence this RFC is generally
not concerned with:

* Issues arising from C++ references. These cannot be kept around when serializing to a binary file stream.
* Performance of the serialization/deserialization.

The general proposal is to implement the pickle protocol using existing Keras saving functionality
as a backend. For example, adding pickle/copy support to Metrics is as simple as:

```python3
class Metric:  # in tf.keras.metrics

    def __reduce_ex__(self, protocol):
        return deserialize, (serialize(metric),)  # where deserialize is tf.keras.metrics.deserialize
```

Documentation for how to use `__reduce__ex__` and other alternatives that allow implementing of
the pickle protocol can be found [here](https://docs.python.org/3/library/pickle.html) and
[here](https://docs.python.org/3/library/copyreg.html) (official Python docs).

For more complex objects (namely `tf.keras.Model`) we can either:

1. Implement a similar approach, but we would need to save the weights and config separately. See [this notebook](https://colab.research.google.com/drive/14ECRN8ZQDa1McKri2dctlV_CaPkE574I?authuser=1#scrollTo=qlXDfJObNXVf) for an example.
2. Use `Model.save` as the backend. This would require implementing support for serializing to memory in `Model.save`, but is overall a better solution (since `Model.save` is the official way to save models in `tf.keras`)

Solution (2) would look something like this (assuming `Model.save` worked with `io.BytesIO()`):

```python3
class Model:

    def __reduce_ex__(self, protocol):
        b = io.BytesIO()  # python built in library
        self.save(b)  # i.e. tf.keras.Model.save
        return load_model, (b.get_value()),)  # where load_model is tf.keras.models.load_model
```

Note how this exactly mirrors the aforementioned blog post regarding PyTorch ([link](https://matthewrocklin.com/blog/work/2018/07/23/protocols-pickle)).

By implementing this in all of Keras' base classes, things will automatically work
with custom metrics and subclassed models.

### Alternatives Considered

The only real alternative is to:

1. Ask all libraries that currently use pickle to make a special case for each Keras object and figure out how each Keras object prefers to be serialized (see use of `serialize` vs `Model.save` above).
2. Ask all users to learn the above as well.
3. Override `__reduce_ex__` to give a user friendly warning instead of failing cryptically.

### Performance Implications

* The performance should be the same as the underlying backend that is already implemented in TF.
* For cases where the user was going to pickle anyway, this will be faster because it uses TF's methods instead of letting Python deal with it naively.
* Tests will consist of running `new_model = pickle.loads(pickle.loads(model))` and then doing checks on `new_model`.

### Dependencies

* Dependencies: does this proposal add any new dependencies to TensorFlow? **NO**
* Dependent projects: are there other areas of TensorFlow or things that use TensorFlow (TFX/pipelines, TensorBoard, etc.) that this affects? **This should not affect anything**

### Engineering Impact

* Do you expect changes to binary size / startup time / build time / test times? **NO**
* Who will maintain this code? Is this code in its own buildable unit? Can this code be tested in its own? Is visibility suitably restricted to only a small API surface for others to use?

This code depends on existing Keras/TF methods. As long as those are maintained and don't break, this code will not break. The new API surface area is very small.

### Platforms and Environments

* Platforms: does this work on all platforms supported by TensorFlow? If not, why is that ok? Will it work on embedded/mobile? Does it impact automatic code generation or mobile stripping tooling? Will it work with transformation tools?
* Execution environments (Cloud services, accelerator hardware): what impact do you expect and how will you confirm?

This will work on anything that is running Python >= 2.7 (as far as I can tell, the pickle protocol has not changed since then).

### Best Practices

* Does this proposal change best practices for some aspect of using/developing TensorFlow? How will these changes be communicated/enforced? **NO**

### Tutorials and Examples

There are plenty of examples of how this can and would be used within all of the issues above, in addition to the linked notebook
([link again](https://colab.research.google.com/drive/14ECRN8ZQDa1McKri2dctlV_CaPkE574I?authuser=1#scrollTo=qlXDfJObNXVf)) which has
end to end implementations and tests for all of this.

### Compatibility

* Does the design conform to the backwards & forwards compatibility [requirements](https://www.tensorflow.org/programmers_guide/version_compat)? **YES**
  
* How will this proposal interact with other parts of the TensorFlow Ecosystem?
      - How will it work with TFLite?  *N/A*
      - How will it work with distribution strategies?  *N/A*
      - How will it interact with tf.function?  *N/A*
      - Will this work on GPU/TPU?  *N/A*
      - How will it serialize to a SavedModel? *Circular question...*

### User Impact

* What are the user-facing changes? How will this feature be rolled out?

We implement it and are done. I do not think there is any need to edit the docs to advertise this feature. We still want
users to use `Model.save` when they are trying to save their model.

## Questions and Discussion Topics

Seed this with open questions you require feedback on from the RFC process.