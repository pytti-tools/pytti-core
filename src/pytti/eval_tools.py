import math
import re
import requests
import io

math_env = None
global_t = 0
global_bands = {}
global_bands_prev = {}
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
        for band in global_bands:
            math_env[band] = global_bands[band]
        if global_bands_prev:
            for band in global_bands_prev:
                math_env[f'{band}_prev'] = global_bands_prev[band]
        try:
            output = eval(string, math_env)
        except SyntaxError as e:
            raise RuntimeError("Error in parametric value " + string)
        eval_memo[string] = output
        return output
    else:
        return string


def set_t(t, band_dict):
    global global_t, global_bands, global_bands_prev, eval_memo
    global_t = t
    if global_bands:
        global_bands_prev = global_bands
    else:
        global_bands_prev = band_dict
    global_bands = band_dict
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
