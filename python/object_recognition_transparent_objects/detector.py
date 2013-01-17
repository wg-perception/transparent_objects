#!/usr/bin/env python
"""
Module defining the transparent objects detector to find objects in a scene
"""

from ecto.blackbox import BlackBoxCellInfo as CellInfo
from object_recognition_core.pipelines.detection import DetectorBase
import ecto
import transparent_objects_cells

########################################################################################################################

class TransparentObjectsDetector(ecto.BlackBox, DetectorBase):

    def __init__(self, *args, **kwargs):
        ecto.BlackBox.__init__(self, *args, **kwargs)
        DetectorBase.__init__(self)

    @classmethod
    def declare_cells(cls, _p):
        return {'main': CellInfo(transparent_objects_cells.Detector)}

    @classmethod
    def declare_forwards(cls, _p):
        return ({'main':'all'}, {'main':'all'}, {'main':'all'})
    
    def connections(self, _p):
        return [self.main]
