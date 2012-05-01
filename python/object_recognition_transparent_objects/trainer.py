#!/usr/bin/env python
"""
Module defining the transparent objects trainer
"""

from object_recognition_core.db.dbtools import db_params_to_db
from object_recognition_core.db.object_db import DbModels, Document, DbDocuments
from object_recognition_core.db.models import Model, find_model_for_object
from object_recognition_core.pipelines.training import TrainingPipeline
from object_recognition_core.utils.json_helper import dict_to_cpp_json_str
import ecto
import transparent_objects_cells

########################################################################################################################

class TransparentObjectsProcessor(ecto.BlackBox):
    """
    """
    _trainer = transparent_objects_cells.Trainer

    def declare_params(self, p):
        p.forward_all('_trainer')
        p.declare('db_models','A list of db docs.')
        
    def declare_io(self, p, i, o):
        o.forward_all('_trainer') # must forward the outputs of our trainer...

    def configure(self, p, i, o):
        self._dealer = ecto.Dealer(tendril=ecto.Tendril(Document()), iterable=p.db_models)
 
    def connections(self):
        return [ self._dealer[:] >> self._trainer[:] ]
       

class TransparentObjectsTrainingPipeline(TrainingPipeline):
    '''Implements the training pipeline functions'''
    @classmethod
    def type_name(cls):
        return "transparent_objects"

    @classmethod
    def processor(cls, *args, **kwargs):
        object_db = kwargs['object_db']
        object_id = kwargs.get('object_id', None)
        submethod = kwargs['submethod']
    
        #db_models = DbModels(db_params, [ object_id ], method, submethod)
        if object_id:
            #TODO these should be loaded from the database?
            json_K = kwargs['pipeline_params']['K']
            json_D = kwargs['pipeline_params']['D']
            imageWidth = kwargs['pipeline_params']['imageWidth']
            imageHeight = kwargs['pipeline_params']['imageHeight']

            document_ids =  find_model_for_object(db_params_to_db(object_db.parameters()), object_id, model_type='mesh')
            print document_ids
            db_models = DbDocuments(object_db, document_ids)
            print 'Found %d meshes:'%len(db_models)
        else:
            #TODO these should be loaded from the database?
            json_K = []
            json_D = []
            imageWidth = 640
            imageHeight = 480

            db_models = []
        return TransparentObjectsProcessor(json_submethod=dict_to_cpp_json_str(submethod), json_K=dict_to_cpp_json_str(json_K), json_D=dict_to_cpp_json_str(json_D), imageWidth=imageWidth, imageHeight=imageHeight, db_models=db_models)

    @classmethod
    def post_processor(cls, *args, **kwarg):
        return transparent_objects_cells.ModelFiller()
