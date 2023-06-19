#Copyright (c) Facebook, Inc. and its affiliates.
#This source code is licensed under the MIT license found in the
#LICENSE file in the root directory of this source tree.

import sys, inspect
# we split the handling between exceptions and errors.
# Errors are defined at critical failures that needs to be debugged by the developerse. Error end with the word "Error"
# Exceptions are caused when the framework results in invalid designs due to careless modifications. These
# are inevitable as not every modification is valid. Exceptions end with the word "Exception"

# ---------------------------
# Error List
# ---------------------------
# a task with no children
class TaskNoChildrenError(Exception):
    pass

# a task with no pe detected
class NoPEError(Exception):
    pass

# a task with no mem detected
class NoMemError(Exception):
    pass

# a task with no bus detected
class NoBusError(Exception):
    pass

class BlockCountDeviationError(Exception):
    pass

#  swap transformation was not executed properly
class IncompleteSwapError(Exception):
    pass

#  design doesn't have a block of a certain type ("pe",  "mem" , "ic")
class NoBlockOfCertainType(Exception):
    pass

#  each Block can only be connected to one bus
class MultiBusBlockError(Exception):
    pass

# bus with no memory was detected
class BusWithNoMemError(Exception):
    pass

# bus with no memory was detected
class SystemICWithPEException(Exception):
    pass


# bus with no PE was detected
class BusWithNoPEError(Exception):
    pass

class NotEnoughIPOfCertainType(Exception):
    pass

#  a block with no tasks mapped to it was detected
class BlockWithNoTaskError(Exception):
    pass

#  IP (accelerators) can not be splitted because they only have on task on them
class IPSplitException(Exception):
    pass


class NoAbException(Exception):
    pass


class TransferException(Exception):
    pass

class RoutingException(Exception):
    pass


#  couldn't find two blocks of the same type to use for cleaning up
class CostPairingException(Exception):
    pass

# ---------------------------
# exception List
# ---------------------------
class MoveNoDesignException(Exception):
    pass


class UnEqualFrontsError(Exception):
    pass


class MoveNotValidException(Exception):
    pass

# ToDO: this exception is for the most part caused by the fact that
# moves (and their corresponding tasks) are determined  before loading of memory.
# Later, we need to fix this, and get rid of this exception (basically make sure whoever calls it, is fixed)
class NoMigrantException(Exception):
    pass

# could not find a task that can run in parallel (for split and migration)
class NoParallelTaskException(Exception):
    pass

# This is a scenario where no migrant is detected to be moved, but it must have
# this is different than NoMigrantException (since the exception scenario is permissable)
class NoMigrantError(Exception):
    pass


# ic migration not supported at the moment
class ICMigrationException(Exception):
    pass

# ic migration not supported at the moment
class NoValidTransformationException(Exception):
    pass

def get_error_classes():
    error_class_list = []
    for name, obj in inspect.getmembers(sys.modules[__name__]):
        if inspect.isclass(obj):
            error_class_list.append(obj)
    return error_class_list

def get_error_names():
    error_name_list = []
    for name, obj in inspect.getmembers(sys.modules[__name__]):
        if inspect.isclass(obj) and "Error" in name:
            error_name_list.append(name)
    return error_name_list

def get_exception_names():
    exception_name_list = []
    for name, obj in inspect.getmembers(sys.modules[__name__]):
        if inspect.isclass(obj) and "Exception" in name:
            exception_name_list.append(name)
    return exception_name_list


errors_names = [el for el in get_error_names()]
exception_names = [el for el in get_exception_names()]

"""
# simple unit test

def foo():
    raise NoPEError
    #raise TaskNoChildrenError
def foo_2():
    return 1/0

try:
    foo_2()
except Exception as e:
    if e.__class__.__name__ in errors_names:
        print("have seend this error before")
    else:
        raise e
"""
