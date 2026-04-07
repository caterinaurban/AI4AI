import onnx
import onnx.numpy_helper

from frontend.mirror import Mirror, Activations


def onnx2mirror(model_path: str) -> Mirror:
    model = onnx.load(model_path)
    graph = model.graph

    initializers = {init.name: onnx.numpy_helper.to_array(init) for init in graph.initializer}

    # Skip Flatten nodes; everything else is Gemm or activation
    nodes = [node for node in graph.node if node.op_type != 'Flatten']

    # Number of inputs from the first Gemm weight matrix
    first_gemm = next(n for n in nodes if n.op_type == 'Gemm')
    W0 = initializers[first_gemm.input[1]]
    transB0 = any(attr.name == 'transB' and attr.i for attr in first_gemm.attribute)
    n_ins = W0.shape[1] if transB0 else W0.shape[0]
    inputs = ["x0%d" % j for j in range(n_ins)]

    activations = {}
    layers = []
    outputs = []

    l = 1
    i = 0
    while i < len(nodes):
        node = nodes[i]
        if node.op_type == 'Gemm':
            W = initializers[node.input[1]]
            b = initializers[node.input[2]]

            transB = any(attr.name == 'transB' and attr.i for attr in node.attribute)
            n_out, n_in = (W.shape[0], W.shape[1]) if transB else (W.shape[1], W.shape[0])

            next_node = nodes[i + 1] if i + 1 < len(nodes) else None
            has_relu = next_node is not None and next_node.op_type == 'Relu'
            has_sigmoid = next_node is not None and next_node.op_type == 'Sigmoid'

            current = {}
            for out_idx in range(n_out):
                lhs = "x%d%d" % (l, out_idx)
                rhs = {}
                for in_idx in range(n_in):
                    w = float(W[out_idx, in_idx]) if transB else float(W[in_idx, out_idx])
                    rhs["x%d%d" % (l - 1, in_idx)] = w
                rhs["_"] = float(b[out_idx])
                current[lhs] = rhs

                if has_relu:
                    activations[lhs] = Activations.RELU
                elif has_sigmoid:
                    activations[lhs] = Activations.SIGMOID
                else:
                    outputs.append(lhs)

            layers.append(current)
            l += 1
            i += 2 if (has_relu or has_sigmoid) else 1
        else:
            i += 1

    return Mirror(inputs, activations, layers, outputs)
