:orphan:

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

References
----------
Ilya Lysenkov, Victor Eruhimov, and Gary Bradski, "`Recognition and Pose Estimation of Rigid Transparent Objects with a Kinect Sensor <http://www.roboticsproceedings.org/rss08/p35.html>`_," 2013 Robotics: Science and Systems Conference (RSS), 2013.

Ilya Lysenkov, and Vincent Rabaud, "Pose Estimation of Rigid Transparent Objects in Transparent Clutter", 2013 IEEE International Conference on Robotics and Automation (ICRA), 2013.
