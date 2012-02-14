ectomodule(transparent_objects_cells
    module.cpp
    train.cpp
    detect.cpp
    db_transparent_objects.cpp
    modelFiller.cpp
    )

find_package(PCL)

link_ecto(transparent_objects_cells
     ${PCL_LIBRARIES}
     ${Boost_LIBRARIES}
     object_recognition_common
     object_recognition_db
     edges_pose_refiner
    )

install_ecto_module(transparent_objects_cells)
