import logging

def setup_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Add a file handler to write logs to training.log
    fh = logging.FileHandler('training.log')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    # Add a stream handler to print logs to the console
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    return logger
