from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

class Logger:
    def __init__(self, path):
        self.path = Path(path)
        # create an instance of SummaryWriter
        self.writer = SummaryWriter(self.path)

    def write(self, epoch=0, stage='Train', **kwargs):
    # record various of loss and distance

        if stage == 'Train':
            for key, value in kwargs.items():
                self.writer.add_scalar('Train/'+str(key), value, epoch)
                self.writer.flush()

        elif stage == 'Valid':
            for key, value in kwargs.items():
                self.writer.add_scalar('Valid/' + str(key), value, epoch)
                self.writer.flush()
        elif 'Test' in stage:
            for key, value in kwargs.items():
                self.writer.add_scalar(stage + str(key), value)
                self.writer.flush()

        else:
            raise ValueError
        return


    def close(self):
        self.writer.close()
        return