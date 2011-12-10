#!/usr/bin/env python
"""
Module defining the transparent objects trainer
"""

from object_recognition.common.utils import dict_to_cpp_json_str
from object_recognition.pipelines.training import TrainingPipeline
from ecto_object_recognition.object_recognition_db import DbModels, Document, DbDocuments
from object_recognition.models import Model, find_model_for_object
from object_recognition.dbtools import db_params_to_db
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

    def processor(self, *args, **kwargs):
        db_params = kwargs['db_params']
        object_id = kwargs['object_id']
        submethod = kwargs['submethod']
    
        #TODO these should be loaded from the database?
        json_K = kwargs['pipeline_params']['K']
        json_D = kwargs['pipeline_params']['D']
    
        #db_models = DbModels(db_params, [ object_id ], method, submethod)
        document_ids =  find_model_for_object(db_params_to_db(db_params), object_id, model_type='mesh')
        print document_ids
        db_models = DbDocuments(db_params, document_ids)
        print 'Found %d meshes:'%len(db_models)
        return TransparentObjectsProcessor(json_submethod=dict_to_cpp_json_str(submethod),json_K=dict_to_cpp_json_str(json_K), json_D=dict_to_cpp_json_str(json_D), db_models=db_models)

    def post_processor(self, *args, **kwarg):
        return transparent_objects_cells.ModelFiller()
