import datetime


class Logger:
    def __init__(self, filename=None):
        self.logfile = filename

    def format_time():
        t = datetime.datetime.now()
        s = t.strftime('%Y-%m-%d %H:%M:%S.%f')
        return s[:-4]

    def log(self, str: str):
        str = f'[{self.format_time()}] {str}'
        print(str)
        if self.logfile is not None:
            with open(self.logfile, mode='a') as f:
                print(str, file=f)
