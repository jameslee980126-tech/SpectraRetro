from typing import Optional

from .utils import USPTODataset
from ..datamodule import DataModule

__all__ = ['USPTO']

class USPTO(DataModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def setup(self, stage: Optional[str] = None):
        self.stage = stage
        if stage == "fit":
            self.train_dataset = USPTODataset(
                stage='train', **self.data_kwargs
            )
            self.valid_dataset = USPTODataset(
                stage='val', **self.data_kwargs
            )
            
        elif stage == "validate":
            self.valid_dataset = USPTODataset(
                stage='val', **self.data_kwargs
            )

        elif stage == "test" or stage == "predict":
            self.test_dataset = USPTODataset(
                stage='test', **self.data_kwargs
            )

        else:
            raise ValueError()

