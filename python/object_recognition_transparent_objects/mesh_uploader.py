#!/usr/bin/env python
import sys
import argparse
import time
import tempfile
import os
import subprocess

import couchdb

import ecto
import object_recognition
from object_recognition import dbtools, models
from tempfile import NamedTemporaryFile

def upload_mesh(db, object_id, cloud_file, mesh_file=None):
    r = models.find_model_for_object(db, object_id, 'mesh')
    m = None
    for model in r:
        m = models.Model.load(db, model)
        print "updating model:", model
        break
    if not m:
        m = models.Model(object_id=object_id, method='mesh', submethod='kinect_fusion')
        print "creating new model."
    m.store(db)
    print m.id
    with open(cloud_file, 'r') as mesh:
        db.put_attachment(m, mesh, filename='cloud.ply')
    if mesh_file:
        with open(mesh_file, 'r') as mesh:
            db.put_attachment(m, mesh, filename='mesh.stl', content_type='application/octet-stream')

FILTER_SCRIPT = '''
<!DOCTYPE FilterScript>
<FilterScript>
 <filter name="Surface Reconstruction: Poisson">
  <Param type="RichInt" value="8" name="OctDepth"/>
  <Param type="RichInt" value="6" name="SolverDivide"/>
  <Param type="RichFloat" value="2" name="SamplesPerNode"/>
  <Param type="RichFloat" value="1" name="Offset"/>
 </filter>
</FilterScript>
'''
FILTER_SCRIPT_PIVOTING = '''
<!DOCTYPE FilterScript>
<FilterScript>
 <filter name="Surface Reconstruction: Ball Pivoting">
  <Param type="RichAbsPerc" value="0.01" min="0" name="BallRadius" max="0.366624"/>
  <Param type="RichFloat" value="50" name="Clustering"/>
  <Param type="RichFloat" value="90" name="CreaseThr"/>
  <Param type="RichBool" value="false" name="DeleteFaces"/>
 </filter>
</FilterScript>
'''
#meshlabserver -i cloud_44ed68c2b66cc8aefc7df45fd63c4ac8_00000.ply -o mug.stl -s meshlab.xml.mlx 

def meshlab(filename_in, filename_out):
    import tempfile, subprocess, os
    f = NamedTemporaryFile(delete=False)
    script = f.name
    f.write(FILTER_SCRIPT)
    f.close()
    subprocess.check_call((['meshlabserver', '-i', filename_in, '-o', filename_out, '-s', script]))
    os.unlink(script)

def simple_mesh_session(dbs, args):
    ply_name = args.ply_file
    stl_name = args.stl_file

    #mesh_name = 'cloud_%s.stl' % args.object_id
    #meshlab(ply_name, mesh_name)

    if args.commit:
        upload_mesh(dbs, args.object_id, ply_name, stl_name)
#        os.unlink(mesh_name)

###################################################################################################################

def parse_args():
    parser = argparse.ArgumentParser(description='Computes a surface mesh of an object from a ply file and upload '
                                     'both to the database')
    parser.add_argument('-o,--object_id', dest='object_id', help='The id of the object', required=True)
    parser.add_argument('-p,--ply_file', dest='ply_file', help='The id of the object', required=True)
    parser.add_argument('-s,--stl_file', dest='stl_file', help='The mesh of the object', required=True)

    object_recognition.dbtools.add_db_arguments(parser)

    args = parser.parse_args()
    return args

if "__main__" == __name__:
    args = parse_args()
    couch = couchdb.Server(args.db_root)
    dbs = dbtools.init_object_databases(couch)

    models.sync_models(dbs)
    simple_mesh_session(dbs, args)
