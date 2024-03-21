""" non grouped utils
"""
import math
import typing
from string import digits

from corl.libraries.units import Quantity, corl_quantity


# TODO FIX TTHIS SHOULD BE A SIMPLE WRAP ON A QUANTITY
def get_wrap_diff_quant(angle0: Quantity, angle1: Quantity, method: int) -> Quantity:
    diff = angle0.to("degree").m - angle1.to("deg").m
    if method == 1:
        # Compute the diff as abs min angle (always positive)
        abs_diff_mod_360 = abs(diff) % 360
        result = 360 - abs_diff_mod_360 if abs_diff_mod_360 > 180 else abs_diff_mod_360
    else:
        # Compute the diff as min angle maintain sign for direction
        result = (diff + 180) % 360 - 180

    return corl_quantity()(result, "degree")


def get_wrap_diff(A: float, B: float, is_rad: bool, method: int) -> float:
    """Returns the min diff angle

        RAD A	         Deg A   RAD A	       Deg B   abs_diff_mod_360	Diff
        1.047197551	      60     2.094395102    120	    60	             60
        2.094395102	     120 	 1.047197551     60	    60	             60
        -2.094395102	-120	-1.047197551    -60	    60	             60
        6.108652382	     350	 0.1745329252    10	   340	             20
        -2.967059728	-170	 2.967059728    170	   340	             20

    Arguments:
        A {float} -- Angle 1 - rad or deg
        B {float} -- Angle 2 - rad or deg

    Returns:
        float -- the min diff angle
    """

    # Convert to degrees if needed.
    temp_a = math.degrees(A) if is_rad else A
    temp_b = math.degrees(B) if is_rad else B

    if method == 1:
        # Compute the diff as abs min angle (always positive)
        abs_diff_mod_360 = abs(temp_a - temp_b) % 360
        result = 360 - abs_diff_mod_360 if abs_diff_mod_360 > 180 else abs_diff_mod_360
    else:
        # Compute the diff as min angle maintain sign for direction
        diff = temp_a - temp_b
        result = (diff + 180) % 360 - 180

    # Return the diff in deg if radians if needed.
    return math.radians(result) if is_rad else result


Str_Tuple_Str = typing.TypeVar("Str_Tuple_Str", str, tuple[str, ...], str | tuple[str, ...])


def replace_magic_strings(s: Str_Tuple_Str, *, platform_name: str | None = None) -> Str_Tuple_Str:
    """Replace magic strings with their value.

    Necessary data is provided as optional keyword arguments.  Therefore, clients can determine
    which keywords they support by which pieces of information they provide.

    Known keywords:
      - %%PLATFORM%% - The name of the current platform
      - %%SIDE%% - The name of the current side, defined as the platform name with digits removed.

    Parameters
    ----------
    s : str, Tuple[str]
        String or tuple of strings in which to make replacements
    platform_name : Optional[str]
        If provided, there is support for the %%PLATFORM%% and %%SIDE%% arguments.
    """

    if platform_name is not None:
        remove_digits = str.maketrans("", "", digits)
        side = platform_name.translate(remove_digits)

        def replace_platform_side(ss: str, platform_name: str, side: str) -> str:
            return ss.replace("%%SIDE%%", side).replace("%%PLATFORM%%", platform_name)

        if isinstance(s, str):
            s = replace_platform_side(s, platform_name, side)
        else:
            s = tuple(replace_platform_side(x, platform_name, side) for x in s)

    return s
