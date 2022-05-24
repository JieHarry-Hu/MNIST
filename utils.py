import time
def print_log(print_string, log):
    print("{}".format(print_string))
    if log is not None:
        log.write('{}\n'.format(print_string))
        log.flush()
def time_for_file():
    ISOTIMEFORMAT = '%d-%h-at-%H-%M-%S'
    return '{}'.format(time.strftime(ISOTIMEFORMAT, time.gmtime(time.time())))