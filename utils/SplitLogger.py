import sys


class Logger(object):
    def __init__(self, target, output_file):
        self.terminal = target
        self.log = open(output_file, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def duplicate_logs(target_path):
    sys.stdout = Logger(sys.stdout, target_path.format("stdout.log"))
    sys.stderr = Logger(sys.stderr, target_path.format("stderr.log"))