# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from tempfile import TemporaryDirectory
from unittest import TestCase

from mmengine.registry import (Registry, count_registered_modules, root,
                               traverse_registry_tree)


class TestUtils(TestCase):

    def test_traverse_registry_tree(self):
        #        Hierarchical Registry
        #                           DOGS
        #                      _______|_______
        #                     |               |
        #            HOUNDS (hound)          SAMOYEDS (samoyed)
        #           _______|_______                |
        #          |               |               |
        #     LITTLE_HOUNDS    MID_HOUNDS   LITTLE_SAMOYEDS
        #     (little_hound)   (mid_hound)  (little_samoyed)
        DOGS = Registry('dogs')
        HOUNDS = Registry('dogs', parent=DOGS, scope='hound')
        LITTLE_HOUNDS = Registry(  # noqa
            'dogs', parent=HOUNDS, scope='little_hound')
        MID_HOUNDS = Registry('dogs', parent=HOUNDS, scope='mid_hound')
        SAMOYEDS = Registry('dogs', parent=DOGS, scope='samoyed')
        LITTLE_SAMOYEDS = Registry(  # noqa
            'dogs', parent=SAMOYEDS, scope='little_samoyed')

        @DOGS.register_module()
        class GoldenRetriever:
            pass

        # traversing the tree from the root
        result = traverse_registry_tree(DOGS)
        self.assertEqual(result[0]['num_modules'], 1)
        self.assertEqual(len(result), 6)

        # traversing the tree from leaf node
        result_leaf = traverse_registry_tree(MID_HOUNDS)
        # result from any node should be the same
        self.assertEqual(result, result_leaf)

    def test_count_all_registered_modules(self):
        temp_dir = TemporaryDirectory()
        results = count_registered_modules(temp_dir.name, verbose=True)
        self.assertTrue(
            osp.exists(
                osp.join(temp_dir.name, 'modules_statistic_results.json')))
        registries_info = results['registries']
        for registry in registries_info:
            self.assertTrue(hasattr(root, registry))
            self.assertEqual(registries_info[registry][0]['num_modules'],
                             len(getattr(root, registry).module_dict))
        temp_dir.cleanup()

        # test not saving results
        count_registered_modules(save_path=None, verbose=False)
        self.assertFalse(
            osp.exists(
                osp.join(temp_dir.name, 'modules_statistic_results.json')))
