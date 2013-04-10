#!/usr/bin/env python
"""
Module defining the transparent objects trainer
"""

from ecto import BlackBoxCellInfo as CellInfo
from object_recognition_core.db import Document, Documents
from object_recognition_core.db.models import find_model_for_object
from object_recognition_core.db.tools import db_params_to_db
from object_recognition_core.pipelines.training import TrainerBase
from object_recognition_core.utils.json_helper import obj_to_cpp_json_str
import ecto
import transparent_objects_cells

########################################################################################################################

class TransparentObjectsProcessor(ecto.BlackBox):
    """
    """
    @staticmethod
    def declare_cells(_p):
        return {'trainer': CellInfo(transparent_objects_cells.Trainer)}

    @staticmethod
    def declare_direct_params(p):
        p.declare('db_models', 'A list of db docs.')

    @staticmethod
    def declare_forwards(_p):
        p = {'trainer': 'all'}
        i = {}
        o = {'trainer': 'all'}
        return (p, i, o)

    def configure(self, p, _i, _o):
        self.dealer = ecto.Dealer(tendril=ecto.Tendril(Document()), iterable=p.db_models)
 
    def connections(self, p):
        return [ self.dealer[:] >> self.trainer[:] ]
       

class TransparentObjectsTrainingPipeline(TrainerBase):
    '''Implements the training pipeline functions'''
    @classmethod
    def type_name(cls):
        return "transparent_objects"

    @classmethod
    def processor(cls, *args, **kwargs):
        object_db = kwargs['object_db']
        object_id = kwargs.get('object_id', None)

        # db_models = Models(db_params, [ object_id ], method, subtype)
        if object_id:
            # TODO these should be loaded from the database?
            json_K = kwargs['pipeline_params']['K']
            json_D = kwargs['pipeline_params']['D']
            imageWidth = kwargs['pipeline_params']['imageWidth']
            imageHeight = kwargs['pipeline_params']['imageHeight']

            document_ids = find_model_for_object(db_params_to_db(object_db.parameters()), object_id, model_type='mesh')
            print document_ids
            db_models = Documents(object_db, document_ids)
            print 'Found %d meshes:' % len(db_models)
        else:
            # TODO these should be loaded from the database?
            json_K = []
            json_D = []
            imageWidth = 640
            imageHeight = 480

            db_models = []
        return TransparentObjectsProcessor(json_subtype=obj_to_cpp_json_str(subtype), json_K=obj_to_cpp_json_str(json_K), json_D=obj_to_cpp_json_str(json_D), imageWidth=imageWidth, imageHeight=imageHeight, db_models=db_models)

    @classmethod
    def post_processor(cls, *args, **kwarg):
        return transparent_objects_cells.ModelFiller()
