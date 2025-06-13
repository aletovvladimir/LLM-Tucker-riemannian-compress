from pathlib import Path

import hydra
import tensorrt as trt
from omegaconf import DictConfig


@hydra.main(config_name='config', config_path='../configs')
def convert_to_trt(config: DictConfig):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    onnx_path = Path(config.tensort.onnx_model_path) / 'tucker_model.onnx'
    
    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            raise ValueError("Failed to parse ONNX model.")

    config_builder = builder.create_builder_config()
    config_builder.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

    profile = builder.create_optimization_profile()
    profile.set_shape("input_ids", (1, 8), (1, 128), (1, 512))
    profile.set_shape("attention_mask", (1, 8), (1, 128), (1, 512))
    config_builder.add_optimization_profile(profile)

    serialized_engine = builder.build_serialized_network(network, config_builder)
    if serialized_engine is None:
        raise RuntimeError("Failed to serialize TensorRT engine.")

    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized_engine)

    output_path = Path(config.tensort.trt_engine_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(engine.serialize())

    print(f"[✓] TensorRT engine saved to {output_path}")


if __name__ == "__main__":
    convert_to_trt()
