class FileListener(object):

    def __init__(self, file_name, header):
        self.file_name = file_name
        self.header = header
        with open(file_name, "w") as file:
            file.write(";".join(header) + "\n")

    def listen(self, **kwargs):
        with open(self.file_name, "a") as file:
            file.write(";".join([str(kwargs.get(col, None)) for col in self.header]) + "\n")
