#!/usr/bin/env python
"""
Module defining the transparent objects detector to find objects in a scene
"""

from ecto_object_recognition_core.object_recognition_core_db import DbModels, ObjectDbParameters
from object_recognition_core.pipelines.detection import DetectionPipeline
from object_recognition_core.utils import json_helper
import transparent_objects_cells

########################################################################################################################

class TransparentObjectsDetectionPipeline(DetectionPipeline):
    import transparent_objects

    @classmethod
    def type_name(cls):
        return 'transparent_objects'

    @classmethod
    def detector(self, *args, **kwargs):
        visualize = kwargs.pop('visualize', False)
        submethod = kwargs.pop('submethod')
        parameters = kwargs.pop('parameters')
        object_ids = parameters['object_ids']
        db_params = ObjectDbParameters(parameters['db'])
        model_documents = DbModels(db_params, object_ids, self.type_name(), json_helper.dict_to_cpp_json_str(submethod))
        registrationMaskFilename = parameters.get('registrationMaskFilename')
        return transparent_objects_cells.Detector(model_documents=model_documents, db_params=db_params,
                                            registrationMaskFilename=registrationMaskFilename, visualize=visualize)
