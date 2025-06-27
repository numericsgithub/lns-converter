import tensorflow as tf
import numpy as np
import re
from typing import List, Callable
import os
import json
from datetime import datetime
import mquat as mq
from typing import List, TypeVar

T = TypeVar("T")

def __vars(obj):
    if hasattr(obj, "__dict__"):
        return vars(obj)
    # it is a list! We have to fake the vars() dict structure
    list_dict = {}
    for index, value in enumerate(obj):
        #list_dict[index] = value
        if value.name in list_dict:
            raise Exception("Model contains duplicate names", value.name)
        list_dict[value.name] = value
    return list_dict

def getAllPropertiesOfType(obj, baseclass:type, searchedclass:type, flatten=True) -> List[object]:
    """
    Finds all properties of type searchedclass in the given obj.
    Also all Fields with type baseclass are searched for properties of type searchedclass recursively.

    Example:
        baseclass = MyClass
        searchedclass = SearchForMeClass

        obj.FieldA : SearchForMeClass
        obj.FieldB : str
        obj.FieldC : MyClass
        obj.FieldC.SubFieldA : MyClass
        obj.FieldC.SubFieldB : SomeOtherClass
        obj.FieldC.SubFieldB.SubSubFieldA : SeachForMeClass
        obj.FieldC.SubFieldA.OtherSubSubFieldA : SeachForMeClass


        This would return three of the four instances of the SeachForMeClass in the obj:
        [[obj, "FieldA"],
         [obj.FieldC, "SubFieldB"],
         [obj.FieldC.SubFieldA, "OtherSubSubFieldA" ]]

         Yes, in the result the tuple [obj.FieldC.SubFieldB, "SubSubFieldA"] is missing!
         The reason is that obj.FieldC.SubFieldB is not of the type MyClass.
         Only Fields with the type MyClass are recursively searched in this example!
    Args:
        obj: The object to search through recursively
        baseclass: All instances with this type are also searched through recursively
        searchedclass: The type to search for
    Returns: A list of tuples. The first element is the parent object. The second element is the property name
    of the Field with the searchedclass

    """
    if hasattr(obj, "getLayers"): # if it is a functional model
        obj = obj.getLayers()
    results = []
    results_not_flatten = []
    for property, value in __vars(obj).items():
        if issubclass(type(value), searchedclass):
            results.append([obj, property])
            results_not_flatten.append([obj, property])
        if issubclass(type(value), baseclass):
            sublist, sublist_not_flat = getAllPropertiesOfType(value, baseclass, searchedclass, flatten=False)
            results.extend(sublist)
            results_not_flatten.append(([obj, property] ,sublist_not_flat))
    if not flatten:
        return results, results_not_flatten
    return results


def filterAllPropertiesByName(prop_list: List[object], or_list: List[str] = [], and_list: List[str] = []) -> List[object]:
    """
    Filters the given prop_list by name using regex.
    At least one of the or_list regex must match with the property name.
    Also, all of the and_list regex must match with the property name.
    Empty lists are skipped. So if both lists are empty no properties are filtered out.

    Example:
        or_list = [".*B.*", ".*C.*"]
        and_list = [".*Sub.*", ".*Field.*"]
        prop_list =
        [[obj, "FieldA"],
         [obj.FieldC, "SubFieldB"],
         [obj.FieldC.SubFieldA, "OtherSubSubFieldA" ]]

        Keep all Properties where the name:
            - contains "B" or "C"  (or_list)
            - contains "Sub" and "Field"  (and_list)
        Resulting in:
        [[obj.FieldC, "SubFieldB"]]
    Args:
        prop_list: List of properties (see return of getAllPropertiesOfType).
        or_list: list of regex strings where AT LEAST ONE has to match the property name.
        and_list: list of regex strings where ALL have to match the property name.

    Returns: List of properties (see return of getAllPropertiesOfType).

    """
    result = []
    if len(or_list) == 0 and len(and_list) == 0:
        return prop_list
    for obj, prop in prop_list:
        element_name = __vars(obj)[prop].name
        all_and_results = [re.search(x, element_name) for x in and_list]
        and_result = True
        for and_result in all_and_results:
            if not and_result:
                and_result = False
                break
        if not and_result:
            continue
        all_or_results = [re.search(x, element_name) for x in or_list]
        or_result = False or len(or_list) == 0
        for or_result in all_or_results:
            if or_result:
                or_result = True
                break
        if or_result:
            result.append([obj, prop])
    return result


def reassignAllProperties(prop_list: List[object], get_new_value_method: Callable[[object], object]):
    """
    Reassigns a new value to all given properties. The new value is derived from the get_new_value_method.
    Args:
        prop_list: List of properties (see return of getAllPropertiesOfType).
        get_new_value_method: a method to call that accepts the old property value as an argument

    Returns: None

    """
    for obj, prop in prop_list:
        newobj = get_new_value_method(__vars(obj)[prop])
        __vars(obj)[prop] = newobj


def setFieldForAll(prop_list: List[object], field_name: str, new_field_value: object):
    """
    sets a new value for all properties with the name == field_name.
    If the property value is a tf.Variable the assign() function is used instead.
    Args:
        prop_list: List of properties (see return of getAllPropertiesOfType).
        field_name: The name of the field
        new_field_value: The new value for all field

    Returns: None

    """
    for obj, prop in prop_list:
        fields = __vars(__vars(obj)[prop]).items()
        was_found = False
        for property, value in fields:
            if property == field_name:
                was_found = True
                if isinstance(__vars(__vars(obj)[prop])[property], tf.Variable):
                    __vars(__vars(obj)[prop])[property].assign(new_field_value)
                else:
                    __vars(__vars(obj)[prop])[property] = new_field_value
                break
        if not was_found:
            raise Exception("When reflecting object", obj, "the property with name \""+str(field_name)+"\" was not found")


def getValues(prop_list) -> List[object]:
    """
    gets all values for all given properties as a list.
    Args:
        prop_list: List of properties (see return of getAllPropertiesOfType).

    Returns: List ob obejcts. The values of all properties.

    """
    for obj, prop in prop_list:
        yield __vars(obj)[prop]
        
def __getAllProps(target, flatten=True):
    return getAllPropertiesOfType(target, tf.keras.layers.Layer, mq.Layer, flatten=flatten)

def __getLoggerProps(target, flatten=True):
    return getAllPropertiesOfType(target, tf.keras.layers.Layer, mq.QuantizerLogger, flatten=flatten)

def __getQuantizerProps(target, flatten=True):
    return getAllPropertiesOfType(target, tf.keras.layers.Layer, mq.QuantizerBase, flatten=flatten)

def __getObjs(target, prop_list):
    return getValues(prop_list)

def __setObjs(target, prop_list, get_new_obj: Callable[[object], object], or_list: List[str] = [], and_list: List[str] = []):
    def enforce_correct_name(oldobj):
        newobj = get_new_obj(oldobj)
        if newobj.name != oldobj.name:
            raise Exception("Reassinging new quantizers, loggers etc. with different names is not allowed! "
                            "old_object.name was \"" + str(oldobj.name) + "\" but new one is \"" + str(newobj.name) + "\"")
        return newobj
    result = prop_list
    result = filterAllPropertiesByName(result, or_list=or_list, and_list=and_list)
    reassignAllProperties(result, enforce_correct_name)
    result_values = __getObjs(target, result)
    return [x.name for x in result_values]

def __setObjsField(target, prop_list, field_name: str, new_field_value: object, or_list: List[str] = [], and_list: List[str] = []):
    result = prop_list
    result = filterAllPropertiesByName(result, or_list=or_list, and_list=and_list)
    setFieldForAll(result, field_name=field_name, new_field_value=new_field_value)
    result_values = __getObjs(target, result)
    return [x.name for x in result_values]


def getAllOfType(target, selected_type:T) -> List[T]:
    return __getObjs(target, getAllPropertiesOfType(target, tf.keras.layers.Layer, selected_type, flatten=True))

def getLoggers(target):
    return __getObjs(target, __getLoggerProps(target))

def setLogger(target, get_new_logger: Callable[[object], object], or_list: List[str] = [], and_list: List[str] = []):
    return __setObjs(target, __getLoggerProps(target), get_new_obj=get_new_logger, or_list=or_list, and_list=and_list)

def setLoggerField(target, field_name: str, new_field_value: object, or_list: List[str] = [], and_list: List[str] = []):
    return __setObjsField(target, __getLoggerProps(target), field_name=field_name, new_field_value=new_field_value, or_list=or_list, and_list=and_list)

def getQuantizers(target):
    return __getObjs(target, __getQuantizerProps(target))

def setQuantizer(target, get_new_quantizer: Callable[[object], object], or_list: List[str] = [], and_list: List[str] = []):
    return __setObjs(target, __getQuantizerProps(target), get_new_obj=get_new_quantizer, or_list=or_list, and_list=and_list)

def setQuantizerField(target, field_name: str, new_field_value: object, or_list: List[str] = [], and_list: List[str] = []):
    return __setObjsField(target, __getQuantizerProps(target), field_name=field_name, new_field_value=new_field_value, or_list=or_list, and_list=and_list)


class __StructuredEntry:
    def __init__(self, is_selected, name, name_selector, obj, prop, depth):
        self.is_selected = is_selected
        self.name = name
        self.name_selector = name_selector
        self.obj = obj
        self.prop = prop
        self.depth = depth


def __getObjSelectionStructue(tree, filtered) -> List[__StructuredEntry]:
    filtered = [__vars(obj)[field_name].name for obj, field_name in filtered]
    def getSubTree(subTree, prev_parent_obj_prop_name="", depth=0):
        result = []
        if type(subTree) is tuple: #We have a new subTree here! #len(subTree) == 2 and isinstance(subTree[1], str):#len(subTree) != 2 or not isinstance(subTree[0], str):
            parent_obj, parent_obj_prop_name = subTree[0]
            subTree_elements = subTree[1]
            for item in subTree_elements:
                result.extend(getSubTree(item, str(prev_parent_obj_prop_name) + "/" + str(parent_obj_prop_name), depth + 1))
        elif len(subTree) == 2:
            obj, prop = subTree
            name_selector = __vars(obj)[prop].name
            name = str(prev_parent_obj_prop_name) + "/" + str(prop)
            newentry = __StructuredEntry(
                is_selected=name_selector in filtered, name=name, name_selector=name_selector,
                obj=obj, prop=prop, depth=depth)
            result.append(newentry)
        return result
    all_results = []
    for parts in tree:
        all_results.extend(getSubTree(parts))
    return all_results

def __printObjSelection(tree, filtered, print_additional_info):
    struct_list = __getObjSelectionStructue(tree, filtered)
    for entry in struct_list:
        pad = "├"
        for i in range(entry.depth):
            pad = "│" + pad
        if entry.is_selected:
            print(pad + '\033[92m' + "»» " + entry.name_selector + '\033[0m', print_additional_info(__vars(entry.obj)[entry.prop]))
        else:
            print(pad + entry.name_selector, print_additional_info(__vars(entry.obj)[entry.prop]))

def __printObjSelectionStructure(tree, filtered, print_additional_info):
    struct_list = __getObjSelectionStructue(tree, filtered)
    for entry in struct_list:
        if entry.is_selected:
            print('\033[92m' + "»» " + entry.name + '\033[0m', print_additional_info(__vars(entry.obj)[entry.prop]))
        else:
            print(entry.name, print_additional_info(__vars(entry.obj)[entry.prop]))

def printQuantizerSelection(target, or_list: List[str] = [], and_list: List[str] = [], print_additional_info=lambda obj:"", print_path_structure=False):
    everything, tree = __getQuantizerProps(target, flatten=False)
    filtered = filterAllPropertiesByName(everything, or_list=or_list, and_list=and_list)
    if print_path_structure:
        __printObjSelectionStructure(tree, filtered, print_additional_info)
    else:
        __printObjSelection(tree, filtered, print_additional_info)
    return tree

def printLoggerSelection(target, or_list: List[str] = [], and_list: List[str] = [], print_additional_info=lambda obj:"", print_path_structure=False):
    everything, tree = __getLoggerProps(target, flatten=False)
    filtered = filterAllPropertiesByName(everything, or_list=or_list, and_list=and_list)
    if print_path_structure:
        __printObjSelectionStructure(tree, filtered, print_additional_info)
    else:
        __printObjSelection(tree, filtered, print_additional_info)
    return tree

def saveLoggers(target, folderpath:str, or_list: List[str] = [], and_list: List[str] = []):
    if folderpath.endswith("/") or folderpath.endswith("\\"):
        folderpath = folderpath[:-1]
    # Save the logs and create all folders
    everything, tree = __getLoggerProps(target, flatten=False)
    filtered = filterAllPropertiesByName(everything, or_list=or_list, and_list=and_list)
    struct_list = __getObjSelectionStructue(tree, filtered)
    if not os.path.exists(folderpath):
        os.makedirs(folderpath)
    for entry in struct_list:
        if entry.is_selected:
            entry_filepath = folderpath + entry.name
            entry_folderpath = os.path.dirname(entry_filepath)
            if not os.path.exists(entry_folderpath):
                os.makedirs(entry_folderpath)
            __vars(entry.obj)[entry.prop].reset(entry_filepath)
            # Log for the folder
            entry_filepath = entry_folderpath + "/" + os.path.splitext(os.path.basename(entry_folderpath))[0]
            if not os.path.exists(entry_filepath):
                with open(entry_filepath + ".meta", "w") as f:
                    meta_data = {}
                    meta_data["class"] = str(type(entry.obj).__name__)
                    meta_data["depth"] = entry.depth
                    meta_data["propname"] = os.path.splitext(os.path.basename(entry_folderpath))[0]
                    meta_data["timestamp"] = datetime.now().strftime("%d.%m.%Y %H:%M:%S")
                    meta_data["name_structured"] = os.path.dirname(entry.name)
                    meta_data["name"] = entry.obj.name
                    f.write(json.dumps(meta_data))
            # Log the logger
            entry_filepath = entry_folderpath + "/" + entry.prop
            with open(entry_filepath + ".meta", "w") as f:
                meta_data = {}
                meta_data["class"] = str(type(__vars(entry.obj)[entry.prop]).__name__)
                meta_data["quantclass"] = str(type(entry.obj).__name__)
                meta_data["depth"] = entry.depth + 1
                meta_data["propname"] = entry.prop
                meta_data["timestamp"] = datetime.now().strftime("%d.%m.%Y %H:%M:%S")
                meta_data["name_structured"] = entry.name
                meta_data["name"] = entry.name_selector
                f.write(json.dumps(meta_data))
    # Add extra meta information for each folder
    everything, tree = __getAllProps(target, flatten=False)
    filtered = filterAllPropertiesByName(everything, or_list=or_list, and_list=and_list)
    struct_list = __getObjSelectionStructue(tree, filtered)
    for entry in struct_list:
        if entry.is_selected:
            entry_filepath = folderpath + entry.name + "/" + os.path.splitext(os.path.basename(entry.name))[0]
            entry_folderpath = os.path.dirname(entry_filepath)
            if not os.path.exists(entry_folderpath):
                continue # If the folder is not already created we can skip it.
                         # It would be there if logs existed for the folder
            with open(entry_filepath + ".meta", "w") as f:
                meta_data = {}
                meta_data["class"] = str(type(__vars(entry.obj)[entry.prop]).__name__)
                meta_data["depth"] = entry.depth
                meta_data["propname"] = entry.prop
                meta_data["timestamp"] = datetime.now().strftime("%d.%m.%Y %H:%M:%S")
                meta_data["name_structured"] = entry.name
                meta_data["name"] = entry.name_selector
                f.write(json.dumps(meta_data))
