from mmengine import MMLogger
import logging


class TestLogger:
    def test_init(self):
        logger = MMLogger.get_instance('mmengine')
        assert logger.name == 'mmengine'
        assert logger.instance_name == 'mmengine'
        # Logger get from `MMLogger.get_instance` does not inherit from
        # `logging.root`
        assert logger.parent is None
        assert len(logger.handlers) == 1
        assert isinstance(logger.handlers[0], logging.StreamHandler)
        assert logger.log_level == logging.NOTSET
        assert 
