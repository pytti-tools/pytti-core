import math
import re
import requests
import io

math_env = None
global_t = 0
global_fLo = 0
global_fMid = 0
global_fHi = 0
eval_memo = {}


def parametric_eval(string, **vals):
    global math_env
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
        math_env["fLo"] = global_fLo
        math_env["fMid"] = global_fMid
        math_env["fHi"] = global_fHi
        try:
            output = eval(string, math_env)
        except SyntaxError as e:
            raise RuntimeError("Error in parametric value " + string)
        eval_memo[string] = output
        return output
    else:
        return string


def set_t(t, fLo, fMid, fHi):
    global global_t, global_fLo, global_fMid, global_fHi, eval_memo
    global_t = t
    global_fLo = fLo
    global_fMid = fMid
    global_fHi = fHi
    eval_memo = {}


def fetch(url_or_path):
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
    """
    Given a string, a regex pattern, and a list of defaults,
    split the string using the regex pattern,
    and return the split string + the defaults

    :param string: The string to be parsed
    :param split: The regex that defines where to split the string
    :param defaults: A list of default values for the tokens
    :return: A list of the tokens.
    """
    tokens = re.split(split, string, len(defaults) - 1)
    tokens = (
        tokens + defaults[len(tokens) :]
    )  # this is just a weird way to backfill defaults.
    return tokens
