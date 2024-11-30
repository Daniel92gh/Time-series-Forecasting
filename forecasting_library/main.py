import time
import argparse
from forecasting_library.tasks.task import TASK
from forecasting_library.utils.logger import logger
from forecasting_library.utils.utils import *


VERSION = "0.0.1"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--version', action='version', version='%(prog)s {version}'.format(version=VERSION),
                        help="print version and exit")
    parser.add_argument('-n', '--model-name', type=str, required=True, 
                        help="model name", choices=["lstm", "regression"])
    parser.add_argument('-t', '--task-type', type=str, required=True, 
                        help="task type", choices=["train", "test"])  
    parser.add_argument('-c', '--config', type=str, required=True,
                        help='model training configuration')      
    parser.add_argument('-d', '--data-dir', type=str, required=True, 
                        help="input data directory")
    parser.add_argument('-w', '--work-dir', default='./work_dir', 
                        type=str, help='working directory')
    parser.add_argument('-o', '--model-dir', default='./model_dir', 
                        type=str, help='model directory')
    return parser.parse_args()


def main():
    start_time = time.time()
    args = parse_args()
    # initialize task
    task = TASK(args.model_name, args.config, args.data_dir, 
                args.work_dir, args.model_dir, args.task_type)
    # run task
    if args.task_type == "train":
        logger.info(f"Training job started for model name : {args.model_name}")
        task.run_train_loop()
    else:
        logger.info(f"Testing job started for model name : {args.model_name}")
        task.run_test_loop()
    end_time = time.time()
    logger.info(f"The task took : {(end_time -start_time)//60} minutes and {(end_time -start_time)%60} seconds")    


if __name__ == "__main__":
    main()    