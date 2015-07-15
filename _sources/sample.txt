Sample
======

You can run a quick sample without using Object Recognition to view how the algorithm works and how it can be used. All necessary data and the source code are located in the ``sample`` directory. The sample is compiled into the ``bin`` directory and you can run it by passing the path to the ``sample`` directory as the command line parameter, for example:

::

  ./bin/sample ../or-transparent-objects/sample/

After a while you should see the input data:

.. figure:: ../../sample/image.png

.. figure:: depth.jpg

and the results of the algorithm working similar to this:

.. figure:: glassSegmentation.jpg

.. figure:: estimatedPoses.jpg
