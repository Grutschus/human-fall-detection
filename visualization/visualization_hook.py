import math
import os.path as osp
from typing import Sequence

from mmaction.engine.hooks import VisualizationHook
from mmaction.registry import HOOKS
from mmaction.structures import ActionDataSample


@HOOKS.register_module()
class CustomVisualizationHook(VisualizationHook):
    """Custom hook that fixes a bug in the original visualization hook."""

    def _draw_samples(
        self,
        batch_idx: int,
        data_batch: dict,
        data_samples: Sequence[ActionDataSample],
        step: int = 0,
    ) -> None:
        """Visualize every ``self.interval`` samples from a data batch.

        Function copied and adopted from
        mmaction.engine.hooks.visualization_hook.VisualizationHook._draw_samples

        Args:
            batch_idx (int): The index of the current batch in the val loop.
            data_batch (dict): Data from dataloader.
            outputs (Sequence[:obj:`ActionDataSample`]): Outputs from model.
            step (int): Global step value to record. Defaults to 0.
        """
        if self.enable is False:
            return

        batch_size = len(data_samples)
        videos = data_batch["inputs"]
        start_idx = batch_size * batch_idx
        end_idx = start_idx + batch_size

        # The first index divisible by the interval, after the start index
        first_sample_id = math.ceil(start_idx / self.interval) * self.interval

        for sample_id in range(first_sample_id, end_idx, self.interval):
            # video.shape [B, C, T, H, W]
            video = videos[sample_id - start_idx]
            B, C, T, H, W = video.shape
            # [C, B, T, H, W]
            video = video.transpose(0, 1)
            # [C, B * T, H, W]
            video = video.reshape(C, B * T, H, W)

            # move channel to the last
            video = video.permute(1, 2, 3, 0).numpy().astype("uint8")

            data_sample = data_samples[sample_id - start_idx]
            if "filename" in data_sample:
                # osp.basename works on different platforms even file clients.
                sample_name = osp.basename(data_sample.get("filename"))
            elif "frame_dir" in data_sample:
                sample_name = osp.basename(data_sample.get("frame_dir"))
            else:
                sample_name = f"visualization/{str(sample_id)}"

            draw_args = self.draw_args
            if self.out_dir is not None:
                draw_args["out_path"] = self.file_client.join_path(
                    self.out_dir, f"{sample_name}_{step}"
                )

            draw_args.pop("show", None)

            self._visualizer.add_datasample(
                sample_name,
                video=video,
                data_sample=data_sample,
                step=step,
                **self.draw_args,
            )
