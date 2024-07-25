# Copyright (c) Facebook, Inc. and its affiliates.
import datetime
import logging
import time
from collections import OrderedDict, abc
from contextlib import ExitStack, contextmanager
from typing import List, Union
import torch
from torch import nn

from detectron2.utils.comm import get_world_size, is_main_process
from detectron2.utils.logger import log_every_n_seconds
from detectron2.evaluation import SemSegEvaluator


class SemSegEvaluatorCustom(SemSegEvaluator):
    def __init__(
        self,
        dataset_name,
        distributed=True,
        output_dir=None,
        palette=None,
        pred_dir=None,
        dist_type=None,
    ):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
            distributed (bool): if True, will collect results from all ranks for evaluation.
                Otherwise, will evaluate the results in the current process.
            output_dir (str): an output directory to dump results.
        """
        super().__init__(
            dataset_name=dataset_name,
            distributed=distributed,
            output_dir=output_dir,
        )

        # update source names
        print(len(self.input_file_to_gt_file))
        self.input_file_to_gt_file_custom = {}
        for src_file, tgt_file in self.input_file_to_gt_file.items():
            assert os.path.basename(src_file).replace('.jpg', '.png') == os.path.basename(tgt_file)
            src_file_custom = os.path.join(pred_dir, os.path.basename(tgt_file))  # output is saved as png
            self.input_file_to_gt_file_custom[src_file_custom] = tgt_file

        color_to_idx = {}
        for cls_idx, color in enumerate(palette):
            color = tuple(color)
            # in ade20k, foreground index starts from 1
            color_to_idx[color] = cls_idx + 1
        self.color_to_idx = color_to_idx
        self.palette = torch.tensor(palette, dtype=torch.float, device="cuda")  # (num_cls, 3)
        self.pred_dir = pred_dir
        self.dist_type = dist_type

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a pipeline.
                It is a list of dicts. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name".
            outputs: the outputs of a pipeline. It is either list of semantic segmentation predictions
                (Tensor [H, W]) or list of dicts with key "sem_seg" that contains semantic
                segmentation prediction in the same format.
        """
        print("processing")
        for input in tqdm.tqdm(inputs):
            # output = output["sem_seg"].argmax(dim=0).to(self._cpu_device)  # chw --> hw
            output = input["file_name"]
            output = Image.open(output)
            output = np.array(output)  # (h, w, 3)
            pred = self.post_process_segm_output(output)
            # use custom input_file_to_gt_file mapping
            gt_filename = self.input_file_to_gt_file_custom[input["file_name"]]
            gt = self.sem_seg_loading_fn(gt_filename, dtype=np.int)

            gt[gt == self._ignore_label] = self._num_classes

            self._conf_matrix += np.bincount(
                (self._num_classes + 1) * pred.reshape(-1) + gt.reshape(-1),
                minlength=self._conf_matrix.size,
            ).reshape(self._conf_matrix.shape)

            if self._compute_boundary_iou:
                b_gt = self._mask_to_boundary(gt.astype(np.uint8))
                b_pred = self._mask_to_boundary(pred.astype(np.uint8))

                self._b_conf_matrix += np.bincount(
                    (self._num_classes + 1) * b_pred.reshape(-1) + b_gt.reshape(-1),
                    minlength=self._conf_matrix.size,
                ).reshape(self._conf_matrix.shape)

            self._predictions.extend(self.encode_json_sem_seg(pred, input["file_name"]))

    def post_process_segm_output(self, segm):
        """
        Post-processing to turn output segm image to class index map

        Args:
            segm: (H, W, 3)

        Returns:
            class_map: (H, W)
        """
        segm = torch.from_numpy(segm).float().to(self.palette.device)  # (h, w, 3)
        # pred = torch.einsum("hwc, kc -> hwk", segm, self.palette)  # (h, w, num_cls)
        h, w, k = segm.shape[0], segm.shape[1], self.palette.shape[0]
        if self.dist_type == 'abs':
            dist = torch.abs(segm.view(h, w, 1, 3) - self.palette.view(1, 1, k, 3))  # (h, w, k)
        elif self.dist_type == 'square':
            dist = torch.pow(segm.view(h, w, 1, 3) - self.palette.view(1, 1, k, 3), 2)  # (h, w, k)
        elif self.dist_type == 'mean':
            dist_abs = torch.abs(segm.view(h, w, 1, 3) - self.palette.view(1, 1, k, 3))  # (h, w, k)
            dist_square = torch.pow(segm.view(h, w, 1, 3) - self.palette.view(1, 1, k, 3), 2)  # (h, w, k)
            dist = (dist_abs + dist_square) / 2.
        else:
            raise NotImplementedError
        dist = torch.sum(dist, dim=-1)
        pred = dist.argmin(dim=-1).cpu()  # (h, w)
        pred = np.array(pred, dtype=np.int)

        return pred



def inference_on_dataset(
    pipeline,
    data_loader,
    evaluator: Union[DatasetEvaluator, List[DatasetEvaluator], None],
    callbacks=None,
):
    """
    Run pipeline on the data_loader and evaluate the metrics with evaluator.
    Also benchmark the inference speed of `pipeline.__call__` accurately.
    The pipeline will be used in eval mode.

    Args:
        pipeline (callable): a callable which takes an object from
            `data_loader` and returns some outputs.

            If it's an nn.Module, it will be temporarily set to `eval` mode.
            If you wish to evaluate a pipeline in `training` mode instead, you can
            wrap the given pipeline and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the pipeline.
        evaluator: the evaluator(s) to run. Use `None` if you only want to benchmark,
            but don't want to do any evaluation.
        callbacks (dict of callables): a dictionary of callback functions which can be
            called at each stage of inference.

    Returns:
        The return value of `evaluator.evaluate()`
    """
    num_devices = get_world_size()
    logger = logging.getLogger(__name__)
    logger.info("Start inference on {} batches".format(len(data_loader)))

    total = len(data_loader)  # inference data loader must have a fixed length
    if evaluator is None:
        # create a no-op evaluator
        evaluator = DatasetEvaluators([])
    if isinstance(evaluator, abc.MutableSequence):
        evaluator = DatasetEvaluators(evaluator)
    evaluator.reset()

    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_data_time = 0
    total_compute_time = 0
    total_eval_time = 0

    with ExitStack() as stack:
        stack.enter_context(torch.no_grad())

        start_data_time = time.perf_counter()
        dict.get(callbacks or {}, "on_start", lambda: None)()
        for idx, inputs in enumerate(data_loader):
            total_data_time += time.perf_counter() - start_data_time
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_data_time = 0
                total_compute_time = 0
                total_eval_time = 0

            start_compute_time = time.perf_counter()
            dict.get(callbacks or {}, "before_inference", lambda: None)()
            outputs = pipeline(inputs)
            dict.get(callbacks or {}, "after_inference", lambda: None)()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time

            start_eval_time = time.perf_counter()
            evaluator.process(inputs, outputs)
            total_eval_time += time.perf_counter() - start_eval_time

            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            data_seconds_per_iter = total_data_time / iters_after_start
            compute_seconds_per_iter = total_compute_time / iters_after_start
            eval_seconds_per_iter = total_eval_time / iters_after_start
            total_seconds_per_iter = (time.perf_counter() - start_time) / iters_after_start
            if idx >= num_warmup * 2 or compute_seconds_per_iter > 5:
                eta = datetime.timedelta(seconds=int(total_seconds_per_iter * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    (
                        f"Inference done {idx + 1}/{total}. "
                        f"Dataloading: {data_seconds_per_iter:.4f} s/iter. "
                        f"Inference: {compute_seconds_per_iter:.4f} s/iter. "
                        f"Eval: {eval_seconds_per_iter:.4f} s/iter. "
                        f"Total: {total_seconds_per_iter:.4f} s/iter. "
                        f"ETA={eta}"
                    ),
                    n=5,
                )
            start_data_time = time.perf_counter()
        dict.get(callbacks or {}, "on_end", lambda: None)()

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
        )
    )

    results = evaluator.evaluate()
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
        
    return results