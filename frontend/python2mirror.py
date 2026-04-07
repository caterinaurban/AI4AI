from frontend.mirror import Mirror, Activations


def python2mirror(file) -> Mirror:
    activations = dict()
    layers = list()

    current = dict()
    assigned = set()
    used = set()
    with open(file, 'r') as model:
        for line in model:
            line = line.strip()

            if line == "" and current:
                layers.append(current)
                current = dict()
                continue
            if line.startswith("#"):
                continue

            if line.startswith("ReLU("):
                var = line[5:-1]
                activations[var] = Activations.RELU
            elif line.startswith("Sigmoid("):
                var = line[8:-1]
                activations[var] = Activations.SIGMOID
            else:
                var, expr = line.split("=", 1)
                var = var.strip()
                expr = expr.strip()
                assigned.add(var)

                parts = expr.split()
                terms = dict()
                i = 0
                while i < len(parts):
                    if "*" in parts[i]:  # term of the form "c*x"
                        coeff, variable = parts[i].split("*")
                        terms[variable] = float(coeff.replace("(", "").replace(")", ""))
                        used.add(variable)
                    elif parts[i].replace(".", "", 1).replace("(", "").replace(")", "").lstrip("-").isdigit():
                        bias = float(parts[i].replace("(", "").replace(")", ""))
                        terms['_'] = bias
                    i += 1

                current[var] = terms

        if current:  # add last layer
            layers.append(current)

    inputs = sorted(list(used - assigned))
    outputs = sorted(list(assigned - used))
    return Mirror(inputs, activations, layers, outputs)
