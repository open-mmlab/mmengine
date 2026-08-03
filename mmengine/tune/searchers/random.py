# Copyright (c) OpenMMLab. All rights reserved.
import random
from typing import Dict

from .searcher import HYPER_SEARCHERS, Searcher


@HYPER_SEARCHERS.register_module()
class RandomSearcher(Searcher):

    def __init__(self, rule: str, hparam_spec: Dict[str, Dict], *args,
                 **kwargs):
        super().__init__(rule, hparam_spec)

    def suggest(self) -> Dict:
        """Suggest a new hparam based on random selection.

        Returns:
            Dict: suggested hparam
        """
        suggestion = {}
        for key, spec in self._hparam_spec.items():
            if spec['type'] == 'discrete':
                suggestion[key] = random.choice(spec['values'])
            elif spec['type'] == 'continuous':
                suggestion[key] = random.uniform(spec['lower'], spec['upper'])

        return suggestion
