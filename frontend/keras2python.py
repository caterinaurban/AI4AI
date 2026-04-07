""" Converter Keras2Python """
import argparse
import os
import keras

from frontend.keras2mirror import keras2mirror
from frontend.mirror2python import mirror2python


def main():
    # loading keras model
    parser = argparse.ArgumentParser()
    parser.add_argument('keras_model', help='model to convert')
    args = parser.parse_args()
    ipt = args.keras_model
    name = os.path.splitext(os.path.basename(ipt))[0]
    model = keras.models.load_model(ipt)
    # converting keras model into mirror representation
    mirror = keras2mirror(model)
    # convertint mirror representation into python program
    mirror2python(mirror, name=name)


if __name__ == '__main__':
    main()
