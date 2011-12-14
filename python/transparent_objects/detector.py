#!/usr/bin/env python
"""
Module defining the transparent objects detector to find objects in a scene
"""

from object_recognition.pipelines.detection import DetectionPipeline
import transparent_objects_cells


########################################################################################################################

class TransparentObjectsDetectionPipeline(DetectionPipeline):
    import transparent_objects

    @classmethod
    def type_name(cls):
        return 'transparent_objects'

    def detector(self, submethod, parameters, db_params, model_documents, args):
        #visualize = args.get('visualize', False)
        #threshold = parameters.get('threshold', 90)
        registrationMaskFilename = parameters.get('registrationMaskFilename')
        return transparent_objects_cells.Detector(model_documents=model_documents, registrationMaskFilename=registrationMaskFilename)
