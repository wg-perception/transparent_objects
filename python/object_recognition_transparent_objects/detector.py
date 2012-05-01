#!/usr/bin/env python
"""
Module defining the transparent objects detector to find objects in a scene
"""

from object_recognition_core.db.object_db import ObjectDb, DbModels
from object_recognition_core.pipelines.detection import DetectionPipeline
from object_recognition_core.utils import json_helper
import transparent_objects_cells

########################################################################################################################

class TransparentObjectsDetectionPipeline(DetectionPipeline):
    @classmethod
    def type_name(cls):
        return 'transparent_objects'

    @classmethod
    def detector(self, *args, **kwargs):
        visualize = kwargs.pop('visualize', False)
        submethod = kwargs.pop('submethod')
        parameters = kwargs.pop('parameters')
        object_ids = parameters['object_ids']
        object_db = ObjectDb(parameters['db'])
        model_documents = DbModels(object_db, object_ids, self.type_name(), json_helper.dict_to_cpp_json_str(submethod))
        registrationMaskFilename = parameters.get('registrationMaskFilename')
        return transparent_objects_cells.Detector(model_documents=model_documents, object_db=object_db,
                                            registrationMaskFilename=registrationMaskFilename, visualize=visualize)
