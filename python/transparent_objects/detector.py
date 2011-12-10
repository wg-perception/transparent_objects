#!/usr/bin/env python
"""
Module defining the transparent objects detector to find objects in a scene
"""

from object_recognition.pipelines.detection import DetectionPipeline

########################################################################################################################

class TransparentObjectsDetectionPipeline(DetectionPipeline):
    import transparent_objects

    @classmethod
    def type_name(cls):
        return 'transparent_objects'

    def detector(self, submethod, parameters, db_params, model_documents, args):
        import transparent_objects
        #visualize = args.get('visualize', False)
        #threshold = parameters.get('threshold', 90)
        return transparent_objects.Detector(model_documents=model_documents)
