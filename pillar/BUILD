py_library(
    name = "tf_util",
    srcs = ["tf_util.py"],
    deps = [
        "//lingvo/tasks/car:detection_decoder",
        "//lingvo/tasks/car/waymo:waymo_metadata",
        "//waymo_open_dataset:label_proto_py_pb2",
        "//waymo_open_dataset/metrics/ops:detection_metrics_ops",
        "//waymo_open_dataset/metrics/ops:py_metrics_ops",
        "//waymo_open_dataset/metrics/python:config_util_py",
        "//waymo_open_dataset/protos:breakdown_proto_py_pb2",
        "//waymo_open_dataset/protos:metrics_proto_py_pb2",
        "//waymo_open_dataset/utils:range_image_utils",
    ],
)

py_library(
    name = "waymo_loader",
    srcs = ["waymo_loader.py"],
    deps = [
        ":tf_util",
        "//pillar/data:waymo_decoder",
    ],
)

py_library(
    name = "loss",
    srcs = ["loss.py"],
    deps = [
    ],
)

py_library(
    name = "network",
    srcs = ["network.py"],
    deps = [
        ":loss",
        ":tf_util",
    ],
)

py_library(
    name = "config",
    srcs = ["config.py"],
    deps = [
    ],
)

py_binary(
    name = "eval",
    srcs = ["eval.py"],
    python_version = "PY3",
    deps = [
        ":config",
        ":network",
        ":tf_util",
        ":waymo_loader",
    ],
)

py_binary(
    name = "train",
    srcs = ["train.py"],
    python_version = "PY3",
    deps = [
        ":config",
        ":network",
        ":tf_util",
        ":waymo_loader",
    ],
)
