import os

import numpy as np


class FileListener(object):

    def __init__(self, file_name, header):
        self.file_name = os.path.join(os.getcwd(), "output", "logs", file_name)
        self.header = header
        with open(self.file_name, "w") as file:
            file.write(";".join(header) + "\n")

    def listen(self, **kwargs):
        with open(self.file_name, "a") as file:
            file.write(";".join([str(kwargs.get(col, None)) for col in self.header]) + "\n")

    def save_best(self, solution):
        np.save(self.file_name.replace("logs", "bests").replace("txt", "npy"), solution)
