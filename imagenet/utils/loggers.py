from tensorboardX import SummaryWriter
import os
from datetime import datetime
import time

class Logger(object):
    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        timestamp = datetime.fromtimestamp(time.time()).strftime('%m%d-%H:%M')
        self.writer = SummaryWriter(os.path.join(log_dir, timestamp))

    def list_of_scalars_summary(self, prefix, tag_value_pairs, step):
        """Log scalar variables."""
        for tag, value in tag_value_pairs:
            self.writer.add_scalar(prefix+'/'+tag, value, step)

