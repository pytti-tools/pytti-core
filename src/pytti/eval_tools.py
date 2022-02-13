import math
import re
import requests
import io

math_env = None
global_t = 0
eval_memo = {}


def parametric_eval(string, **vals):
    '''
    Evaluates a string as a mathematical expression. Only functions available in pythons native
    "math" library are supported.
    
    :param string: The string to be evaluated
    :return: A function that takes a single argument, t, and returns the value of the parametric
    expression at that t.
    '''
    global math_env # what is "math_env"?
    if string in eval_memo:
        return eval_memo[string]
    if isinstance(string, str):
        if math_env is None:
            math_env = {
                "abs": abs,
                "max": max,
                "min": min,
                "pow": pow,
                "round": round,
                "__builtins__": None,
            }
            math_env.update(
                {key: getattr(math, key) for key in dir(math) if "_" not in key}
            )
        math_env.update(vals)
        math_env["t"] = global_t
        try:
            output = eval(string, math_env)
        except SyntaxError as e:
            raise RuntimeError("Error in parametric value " + string)
        eval_memo[string] = output
        return output
    else:
        return string


def set_t(t):
    '''
    Set the global_t variable to the value of t, and reset the global eval_memo
    
    :param t: the current time step
    '''
    global global_t, eval_memo
    global_t = t
    eval_memo = {}


def fetch(url_or_path):
    '''
    Takes a URL or path to a file,
    fetches it, and returns a file descriptor
    
    :param url_or_path: The URL or path to the file you want to download
    :return: A file descriptor.
    '''
    if str(url_or_path).startswith("http://") or str(url_or_path).startswith(
        "https://"
    ):
        r = requests.get(url_or_path)
        r.raise_for_status()
        fd = io.BytesIO()
        fd.write(r.content)
        fd.seek(0)
        return fd
    return open(url_or_path, "rb")


def parse(string, split, defaults):
    '''
    Given a string, a regex pattern, and a list of defaults, 
    split the string using the regex pattern, 
    and return the list of tokens with the defaults at the end
    
    :param string: The string to be parsed
    :param split: The regex that defines where to split the string
    :param defaults: A tuple of default values for the last few parameters
    :return: A list of the tokens.
    '''
    tokens = re.split(split, string, len(defaults) - 1)
    tokens = tokens + defaults[len(tokens) :]
    return tokens
