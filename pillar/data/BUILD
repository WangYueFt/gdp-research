py_binary(
    name = "generate_waymo_dataset",
    srcs = ["generate_waymo_dataset.py"],
    python_version = "PY3",
    deps = [
        ":waymo_decoder",
        "//waymo_open_dataset:dataset_proto_py_pb2",
    ],
)

py_binary(
    name = "waymo_decoder",
    srcs = ["waymo_decoder.py"],
    python_version = "PY3",
    deps = [
        "//waymo_open_dataset:dataset_proto_py_pb2",
        "//waymo_open_dataset/utils:range_image_utils",
        "//waymo_open_dataset/utils:transform_utils",
    ],
)
