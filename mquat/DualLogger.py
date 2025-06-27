# -*- coding: utf-8 -*-

import sys

# logging class that writes output to the console and a specified file
class DualLogger:
    def __init__(self, log_file):
        self.log_file = log_file
        self.stdout = sys.stdout

    def write(self, data):
        self.log_file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.log_file.flush()
        self.stdout.flush()
        
# variable to restore the old stdout after logging stops
old_stdout = None
# helper function to start logging
def start_logging(log_file):
    global old_stdout
    old_stdout = sys.stdout
    sys.stdout = DualLogger(log_file)
    
# helper function to stop logging
def stop_logging():
    if old_stdout != None:
        sys.stdout.flush()
        sys.stdout = old_stdout