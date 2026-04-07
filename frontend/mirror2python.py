from frontend.mirror import Activations


def mirror2python(model, name='model'):
    otp = name + '.py'
    with open(otp, "w") as pythoncode:
        print("", file=pythoncode)
        for layer in model.layers:
            for out, expr in layer.items():
                print("%s =" % out, end=" ", file=pythoncode)
                bias = ""
                for ins, weight in expr.items():
                    if ins == '_':
                        bias = "(%f)" % weight
                    else:
                        print("(%f)*%s +" % (weight, ins), end=" ", file=pythoncode)
                print(bias, file=pythoncode)
            print("#", file=pythoncode)
            for activation, typ in model.activations.items():
                if activation in layer:
                    if typ == Activations.RELU:
                        print("ReLU(%s)" % activation, file=pythoncode)
                    else:
                        assert typ == Activations.SIGMOID
                        print("Sigmoid(%s)" % activation, file=pythoncode)
            print("", file=pythoncode)
