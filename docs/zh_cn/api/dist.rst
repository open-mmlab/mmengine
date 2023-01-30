.. role:: hidden
    :class: hidden-section

mmengine.dist
===================================

.. contents:: mmengine.dist
   :depth: 2
   :local:
   :backlinks: top

.. currentmodule:: mmengine.dist

dist
----------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   gather
   gather_object
   all_gather
   all_gather_object
   all_reduce
   all_reduce_dict
   all_reduce_params
   broadcast
   sync_random_seed
   broadcast_object_list
   collect_results
   collect_results_cpu
   collect_results_gpu

utils
----------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   get_dist_info
   init_dist
   init_local_group
   get_backend
   get_world_size
   get_rank
   get_local_size
   get_local_rank
   is_main_process
   master_only
   barrier
   is_distributed
   get_local_group
   get_default_group
   get_data_device
   get_comm_device
   cast_data_device
