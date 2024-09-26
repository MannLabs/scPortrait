# helper functions for paralellization
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm.auto import tqdm


def execute_indexed_parallel(func: Callable, *, args: list, tqdm_args: dict = None, n_threads: int = 10) -> list:
    """parallelization of function call with indexed arguments using ThreadPoolExecutor. Returns a list of results in the order of the input arguments.

    Args:
        func (Callable): _description_
        args (list): _description_
        tqdm_args (dict, optional): _description_. Defaults to None.
        n_threads (int, optional): _description_. Defaults to 10.

    Returns:
        list: containing the results of the function calls in the same order as the input arguments
    """
    if tqdm_args is None:
        tqdm_args = {"total": len(args)}
    elif "total" not in tqdm_args:
        tqdm_args["total"] = len(args)

    results = [None for _ in range(len(args))]
    with ThreadPoolExecutor(n_threads) as executor:
        with tqdm(**tqdm_args) as pbar:
            futures = {executor.submit(func, *arg): i for i, arg in enumerate(args)}
            for future in as_completed(futures):
                index = futures[future]
                results[index] = future.result()
                pbar.update(1)

    return results


def execute_parallel(func: Callable, *, args: list, tqdm_args: dict = None, n_threads: int = 10) -> None:
    """parallelization of function call with ThreadPoolExecutor.

    Args:
        func (Callable): _description_
        args (list): _description_
        tqdm_args (dict, optional): _description_. Defaults to None.
        n_threads (int, optional): _description_. Defaults to 10.

    Returns:
        None
    """
    if tqdm_args is None:
        tqdm_args = {"total": len(args)}
    elif "total" not in tqdm_args:
        tqdm_args["total"] = len(args)

    with ThreadPoolExecutor(n_threads) as executor:
        with tqdm(**tqdm_args) as pbar:
            futures = {executor.submit(func, *arg): i for i, arg in enumerate(args)}
            for _ in as_completed(futures):
                pbar.update(1)

    return None
