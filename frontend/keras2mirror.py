from frontend.mirror import Mirror, Activations


def keras2mirror(model) -> Mirror:
    dense_layers = [layer for layer in model.layers if len(layer.get_weights()) >= 2]
    n_ins = dense_layers[0].get_weights()[0].shape[0]
    inputs = ["x0%d" % j for j in range(0, n_ins)]
    activations = dict()
    outputs = list()
    #
    layers = list()
    l = 1
    for layer in dense_layers:
        weights = layer.get_weights()[0]
        biases = layer.get_weights()[1]
        ins = weights.shape[0]
        outs = weights.shape[1]
        #
        current = dict()
        for i in range(0, outs):
            lhs = "x%d%d" % (l, i)
            rhs = dict()
            for j in range(0, ins):
                rhs["x%d%d" % (l - 1, j)] = weights[j][i]
            rhs["_"] = biases[i]
            current[lhs] = rhs
            #
            activation = layer.get_config().get('activation', None)
            if activation == 'relu':
                activations[lhs] = Activations.RELU
            elif activation == 'sigmoid':
                activations[lhs] = Activations.SIGMOID
            else:   # output layer
                assert activation == 'linear'
                outputs.append(lhs)
        #
        layers.append(current)
        l = l + 1
    return Mirror(inputs, activations, layers, outputs)
