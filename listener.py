import os

import numpy as np


class FileListener(object):

    def __init__(self, file_name, header):
        self.file_name = file_name
        self.header = header
        with open(self.get_log_file_name(file_name), "w") as file:
            file.write(";".join(header) + "\n")

    def listen(self, **kwargs):
        with open(self.get_log_file_name(self.file_name), "a") as file:
            file.write(";".join([str(kwargs.get(col, None)) for col in self.header]) + "\n")

    def save_best(self, solution):
        np.save(self.get_best_file_name(self.file_name), solution)

    @classmethod
    def get_log_file_name(cls, file_name):
        return ".".join([os.path.join(os.getcwd(), "output", "ablation", "logs", file_name), "txt"])

    @classmethod
    def get_best_file_name(cls, file_name):
        return ".".join([os.path.join(os.getcwd(), "output", "ablation", "bests", file_name), "npy"])
