import os 
import logging

current_dir = os.path.dirname(os.path.abspath(__file__))
log_path = os.path.abspath(os.path.join(current_dir, '..', '..', 'log'))

# create a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# create a file handler
file_handler = logging.FileHandler(log_path, mode='w')
file_handler.setLevel(logging.INFO)

# create a console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# define the formatter
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# add the handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)
