from mmaction.registry import TRANSFORMS  # type: ignore
from mmcv.transforms import BaseTransform  # type: ignore

import pandas as pd
import numpy as np

@TRANSFORMS.register_module()
class RandomUnderSampling(BaseTransform):
    """Randomly drop samples from a given classes."""

    def __init__(self, drop_ratios: list(float), to_reduce_classes:list(str)) -> None:
        self.drop_ratio = drop_ratios
        self.to_reduce_classes=to_reduce_classes

    def transform(self, results: dict) -> dict:
        """Perform the `RandomUnderSampling` transformation.

        Args:
            results (dict): The result dict.

        Returns:
            dict: The result dict.
        """
        #ToDo: exceptions
        df= pd.DataFrame.from_dict(results)

        for (ratio,reduce_class) in zip(self.drop_ratios,self.to_reduce_classes):
            class_indices= df.index[df['label']==reduce_class].tolist()
            if len(class_indices):
                to_drop_indices=np.random.choice(class_indices,size=np.ceil(ratio* len(class_indices))).astype(int)
                df=df.drop(to_drop_indices)
        results= df.to_dict() 
        return  results


