Algorithm Details
=================

Training
--------

Model Construction
^^^^^^^^^^^^^^^^^^
This algorithm needs a predefined 3D model of the transparent object you want to detect. 
So you need to provide your model, either by using KinectFusion to construct one or downloading from web.

Silhouettes Generation
^^^^^^^^^^^^^^^^^^^^^^
Once you have the model, the training algorithm will rotate the 3D model in different viewpoint, and store the silhouette of each viewpoint in database.(The silhouettes are different in different viewpoints)
This is because in testing stage, the test silhouette (the silhouette in the scene) can be matched to the silhouettes in database(training silhouettes). The best match provide hint about the pose of test silhouette.

Testing
-------

Test Silhouette Detection
^^^^^^^^^^^^^^^^^^^^^^^^^
In testing stage, first the depth map from Kinect is used to search the candidate region of transparency. The idea is that transparent object cause NaN in depth map(black region in depth map), so the NaN region in depth map can be candidate of transparency. 
Once the candidate region are fetched, GrabCut is used to crop the test silhouette.

Pose Estimation(Estimate Rx, Ry, Rz and T)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The test silhouette is now at hand, it can be matched with the silhouettes in the database by first applying Proscrutes Analysis to find similarity transform of each training silhouette to test silhouette. (In this step, Rx&Ry are computed)
The next step is to find Rz and T by applying weak projection model. After this step, Rx, Ry, Rz and T of each training silhouette are all computed.
Finally, Chamfer matching is used to score each training silhouette and a rough pose estimation can thus be computed.
