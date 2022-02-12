class FileListener(object):

    def __init__(self, file_name):
        self.file_name = file_name
        with open(file_name, "w") as file:
            file.write(";".join(["iteration", "best.fitness"]) + "\n")

    def listen(self, **kwargs):
        with open(self.file_name, "a") as file:
            file.write(";".join([str(kwargs["iteration"]), str(kwargs["best.fitness"])]) + "\n")
