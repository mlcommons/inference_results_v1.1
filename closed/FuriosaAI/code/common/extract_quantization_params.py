import onnx
import struct
for model in 'mlcommons_resnet50_v1.5_int8.onnx', 'mlcommons_ssd_mobilenet_v1_int8.onnx', 'mlcommons_ssd_resnet34_int8.onnx':
    print(model)
    m = onnx.load(model)
    input_names = set(x.name for x in m.graph.input)
    output_names = set(x.name for x in m.graph.output)
    inner_names = set(x.name for x in m.graph.initializer)
    tensors = {x.name:x for x in m.graph.initializer}

    input_names -= inner_names
    output_names -= inner_names


    q = {}
    for x in m.graph.quantization_annotation:
        if x.tensor_name in input_names or x.tensor_name in output_names:
            d = {kv.key:kv.value for kv in x.quant_parameter_tensor_names}
            s = struct.unpack('f', tensors[d['SCALE_TENSOR']].raw_data)[0]
            z = struct.unpack('b', tensors[d['ZERO_POINT_TENSOR']].raw_data)[0]
            q[x.tensor_name] = dict(s=s, z=z)

    # x = s(q-z)

    print()
    print('  Inputs:', len(input_names))
    for i in input_names:
        print('    ',i, q[i])
    print('  Outputs:', len(output_names))
    for o in output_names:
        print('    ',o, q[o])
    print()
