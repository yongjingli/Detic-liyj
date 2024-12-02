import argparse
import os
import torch
import sys
from pathlib import Path
from torch import Tensor, nn

import detectron2.data.transforms as T
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader, detection_utils
from detectron2.export import TracingAdapter, dump_torchscript_IR
from detectron2.modeling import GeneralizedRCNN, RetinaNet, build_model
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.projects.point_rend import add_pointrend_config
from detectron2.structures import Boxes
from detectron2.utils.env import TORCH_VERSION
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import setup_logger

FILE = Path(__file__).resolve()
# ROOT = FILE.parents[0] / 'Detic'  # Detic root directory
ROOT = FILE.parents[1]  # Detic root directory
CONFIGS = ROOT / 'configs'

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

sys.path.insert(0, str(ROOT / 'third_party/CenterNet2/'))

os.chdir(ROOT)

from centernet.config import add_centernet_config
from detic.config import add_detic_config


def setup_cfg(args):
    cfg = get_cfg()
    # cuda context is initialized before creating dataloader, so we don't fork anymore
    cfg.DATALOADER.NUM_WORKERS = 0
    add_pointrend_config(cfg)
    add_centernet_config(cfg)
    add_detic_config(cfg)
    cfg.merge_from_file(args.config_file)
    print(args.opts)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


# experimental. API not yet final
def export_tracing(torch_model, inputs):
    assert TORCH_VERSION >= (1, 8)
    image = inputs[0]["image"]
    inputs = [{"image": image}]  # remove other unused keys

    if isinstance(torch_model, GeneralizedRCNN):

        def inference(model, inputs):
            # use do_postprocess=False so it returns ROI mask
            inst = model.inference(inputs, do_postprocess=False)[0]
            return [{"instances": inst}]

    else:
        inference = None  # assume that we just call the model directly

    traceable_model = TracingAdapter(torch_model, inputs, inference)
    traceable_model.eval()

    if args.format == "torchscript":
        ts_model = torch.jit.trace(traceable_model, (image,))
        with PathManager.open(os.path.join(args.output, "model.ts"), "wb") as f:
            torch.jit.save(ts_model, f)
        dump_torchscript_IR(ts_model, args.output)
    elif args.format == "onnx":
        with PathManager.open(os.path.join(args.output, "model.onnx"), "wb") as f:
            torch.onnx.export(traceable_model, (image,), f, opset_version=16)
    logger.info("Inputs schema: " + str(traceable_model.inputs_schema))
    logger.info("Outputs schema: " + str(traceable_model.outputs_schema))


def get_sample_inputs(args):
    if args.sample_image is None:
        # get a first batch from dataset
        data_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0])
        first_batch = next(iter(data_loader))
        return first_batch
    else:
        # get a sample data
        original_image = detection_utils.read_image(args.sample_image, format=cfg.INPUT.FORMAT)
        # Do same preprocessing as DefaultPredictor
        aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )
        height, width = original_image.shape[:2]
        image = aug.get_transform(original_image).apply_image(original_image)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

        inputs = {"image": image, "height": height, "width": width}

        # Sample ready
        sample_inputs = [inputs]
        return sample_inputs


if __name__ == "__main__":
    # python export_model.py --config-file $detic_config_dir/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml  --output ./output  --export-method tracing --format torchscript/onnx --sample-image ./00001.jpg MODEL.WEIGHTS $detic_ckpt_dir/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth MODEL.DEVICE cuda
    parser = argparse.ArgumentParser(description="Export a model for deployment.")
    parser.add_argument(
        "--format",
        choices=["onnx", "torchscript"],
        help="output format",
        default="onnx",
    )
    parser.add_argument(
        "--export-method",
        choices=["tracing"],
        help="Method to export models",
        default="tracing",
    )
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument("--sample-image", default=None, type=str, help="sample image for input")
    parser.add_argument("--output", help="output directory for the converted model")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    logger = setup_logger()
    logger.info("Command line arguments: " + str(args))
    PathManager.mkdirs(args.output)

    cfg = setup_cfg(args)
    print(cfg)

    # create a torch model
    torch_model = build_model(cfg)
    DetectionCheckpointer(torch_model).resume_or_load(cfg.MODEL.WEIGHTS)

    # get sample data
    sample_inputs = get_sample_inputs(args)

    assert TORCH_VERSION >= (1, 8)
    image = sample_inputs[0]["image"]
    inputs = [{"image": image}]  # remove other unused keys

    # convert and save model
    if args.export_method == "tracing":
        exported_model = export_tracing(torch_model, sample_inputs)