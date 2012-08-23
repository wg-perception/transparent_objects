.. _transparent_objects:

object_recognition_transparent_objects: Recognition of Transparent Objects
==========================================================================

Transparent objects is a pipeline that can detect and estimate poses of transparent objects, given a point cloud model of an object. The pipeline if fully integrated into the Recognition Kitchen so usual training and detection from Object Recognition can be used with subsequent grasping. See ROS Quick Guide or Object Recognition for details how to use it.

Sample
------
You can run a quick sample without using Object Recognition to view how the algorithm works and how it can be used. All necessary data and the source code are located in the ``sample`` directory. The sample is compiled into the ``bin`` directory and you can run it by passing the path to the ``sample`` directory as the command line parameter, for example:

::

  ./bin/sample ../or-transparent-objects/sample/

After a while you should see the input data:

.. figure:: ../../sample/image.png

.. figure:: depth.jpg

and the results of the algorithm working similar to this:

.. figure:: glassSegmentation.jpg

.. figure:: estimatedPoses.jpg

