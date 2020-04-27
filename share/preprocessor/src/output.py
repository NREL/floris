
import json

class Output():
    """
    Output is a helper class for generating output files.
    """
    def __init__(self, filename):
        self.filename = filename
        self.file = open(self.filename, "w")
        self.ln = "\n"

    def write_empty_line(self):
        self.write_line("")

    def write_line(self, line):
        self.file.write(line + self.ln)

    def end(self):
        self.file.close()
    
    def write_dictionary(self, dictionary):
        self.file.write(
            json.dumps(
                dictionary,
                indent=2,
                # sort_keys=True
            )
        )