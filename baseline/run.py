import os
import sys
import time
import json

import argparse
import numpy as np
import traceback as tr
from pathlib import Path
from datetime import datetime
from typing import Tuple, List, Union, Iterable, Optional, Dict, Any
from huggingface_hub.utils import HFValidationError

sys.path.append(str(Path(__file__)))
os.chdir(os.path.dirname(os.path.realpath(__file__)))
print(os.getcwd())


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)
    

class InvalidAnswerType(Exception):
    """
    Raised when the type of the answer is wrong.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self,
                 func_name: str,
                 ret_type: str,
                 message: Optional[str] = None):
        self.message = message if message is not None else f"The answer returned from `{func_name}` function should be of type `{ret_type}`. Please, check the returned value type."
        super().__init__(self.message)


def check_errors(errors: Union[list, str, dict], init_path: str):
    """
    Check whether any errors during evaluation occurred.
    :param errors: a dict/list/str of error logs;
    :param init_path: a path to save log;
    :return: None
    """
    if errors:
        if not os.path.exists(init_path):
            os.makedirs(init_path)
        with open(f'{init_path}error.json', 'w') as fp:
            json.dump(errors, fp, ensure_ascii=False)
        exit("Error is written to error.json")

# User code imports
try:
    from team_code.evaluate import setup_model_and_tokenizer, process_data_sample, evaluate_generative, evaluate_ppl
except ImportError as e:
    exception_det = f"Please, provide the evaluation functions that can be imported from the `team_code`, and check the availability of the imported modules and libraries.\n{e}"
    print("ImportError: " + exception_det)
    errors = {"exception": exception_det}
    check_errors(errors, "./")


def read_data(path: str) -> List[Dict[str, Any]]:
    """
    Reading evaluation test dataset.
    :param path: path to the file;
    :return: data (List[Dict[str, Any).
    """
    try:
        with open(path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError("Data file was not found by given path `{path}`, check the path correctness.")
    return data


def launch_evaluation(dataset: List[Dict[str, Any]], folder: Optional[str] = "./"):
    errors = {}
    try:
        model, processor, tokenizer = setup_model_and_tokenizer()

        for_generative_metrics = {}
        for_classification_metrics = {}
        for sample_index, sample in enumerate(dataset):
            proc_sample = process_data_sample(sample, processor,
                                              modality_type='video')  # currently run on video tasks
            if os.path.exists(sample.get("audio", '')):
                audio_proc_sample = process_data_sample(sample, processor,
                                                        modality_type='audio')
                proc_sample.update(audio=audio_proc_sample['audio'])
            if proc_sample['task_type'] == 'qa':
                prediction = evaluate_ppl(proc_sample, model, tokenizer, modality_type='video')

                # Check if response type - should be int/float
                if prediction is None:
                    raise InvalidAnswerType(
                        'evaluate_ppl',
                        "None",
                        "The answer returned from `evaluate_ppl` function is None. Please, check the this function.")

                if not isinstance(prediction, int | float):
                    raise InvalidAnswerType('evaluate_ppl', 'int/float')

                for_classification_metrics[sample_index] = prediction

            elif proc_sample['task_type'] == 'captioning':
                prediction = evaluate_generative(proc_sample, model, tokenizer, modality_type='video')

                # Check if response type - should be string
                if prediction is None:
                    raise InvalidAnswerType(
                        'evaluate_generative',
                        'None',
                        "The answer returned from `evaluate_generative` function is None. Please, check the this function.")

                if not isinstance(prediction, str):
                    raise InvalidAnswerType('evaluate_ppl', 'string')

                for_generative_metrics[sample_index] = prediction

            else:
                print(f"Unknown request type: `{proc_sample['task_type']}` in data sample #{sample_index}.")
                continue
    except FileNotFoundError as e:
        error = tr.TracebackException(exc_type =type(e),exc_traceback = e.__traceback__ , exc_value =e).stack[-1]
        exception_det = '{} in {} row'.format(e, error.lineno)
        print("FileNotFoundError: " + exception_det)
        errors['exception'] = f"FileNotFoundError: Some problem with file's path occurred.\nCheck the correct path for the models' weights."
        return errors
    except HFValidationError as e:
        error = tr.TracebackException(exc_type =type(e),exc_traceback = e.__traceback__ , exc_value =e).stack[-1]
        exception_det = '{} in {} row'.format(e, error.lineno)
        print("HFValidationError: " + exception_det)
        errors['exception'] = f"HFValidationError: Some problem with downloading hf checkpoints.\nPlease, remember that there is no Internet access from the job and all the files (including models' weights) should be either uploaded within the docker image, or the submission zip."
        return errors
    except IndexError as e:
        print(f"IndexError: {str(e)}")
        errors['exception'] = f"IndexError: Check the correctness of the indices."
        return errors
    except TypeError as e:
        print(f"TypeError: {str(e)}")
        errors['exception'] = f"TypeError: Check the correctness of the data types."
        return errors
    except ValueError as e:
        print(f"ValueError: {str(e)}")
        errors['exception'] = f"ValueError: This error is probably (but not necessarily) a result of wrong returned values, please, check whether all values are returned correctly."
        return errors
    except InvalidAnswerType as e:
        exception_det = e.message
        print("ValueError: " + exception_det)
        errors['exception'] = f"InvalidAnswerTypeError: {exception_det}"
        return errors
    except ImportError as e:
        print(f"ImportError: {str(e)}")
        errors['exception'] = f"ImportError: Check the availability of the imported modules and libraries."
        return errors
    except ModuleNotFoundError as e:
        print(f"ModuleNotFoundError: {str(e)}")
        errors['exception'] = f"ImportError: Check the availability of the imported modules and libraries."
        return errors
    except Exception as e:
        error = tr.TracebackException(exc_type =type(e),exc_traceback = e.__traceback__ , exc_value =e).stack[-1]
        exception_det = '{} in {} row'.format(e, error.lineno)
        print(f"Exception: {exception_det}")
        errors['exception'] = f"Exception: Some error occurred during the run, please refer to the logs."
        return errors

    # Save results to predefined dir
    # IMPORTANT: the filename should be exactly the same as in example (see submission description for more details).
    # for classification metric filename and path: './output_path_from_job/for_classification_metric.json'
    # for generative metric filename and path: './output_path_from_job/for_generative_metric.json'
    with open(os.path.join(folder, 'for_classification_metric.json'), mode="w") as f:
        json.dump(for_classification_metrics, f)
    with open(os.path.join(folder, 'for_generative_metric.json'), mode="w") as f:
        json.dump(for_generative_metrics, f)

    return None



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=False,
                        help="path to the file with input data",
                        default="./data/dataset.json")
    args = parser.parse_args()

    dataset_filename = args.input_path
    dataset = read_data(dataset_filename)

    disable_torch_init()

    for_metric_folder = "./output_path_from_job"
    os.makedirs(for_metric_folder, exist_ok=True)

    # Video Inference
    start_time = datetime.now()
    errors = launch_evaluation(dataset, for_metric_folder)
    check_errors(errors, "./")
    print(datetime.now() - start_time)
