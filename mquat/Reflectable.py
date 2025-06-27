
import mquat.Reflection as rf
import os
import re
from typing import List, Callable, TypeVar

T = TypeVar("T")

class Reflectable:
    def __init__(self):
        pass

    def getLoggers(self):
        """
         gets all loggers.

         Returns: List of loggers

         """
        return rf.getLoggers(self)

    def setLogger(self, get_new_logger: Callable[[object], object], or_list: List[str] = [], and_list: List[str] = []):
        """
        Sets all selected loggers to a new logger instance. The instance is received from the get_new_logger function.

        Args:
            get_new_logger: a function that accepts the old logger instance and retruns a new one.
            IMPORTANT the new instance.name has to match the old instance.name
            or_list: at least one of the regex have to match the logger.name
            and_list: all of the regex have to match the logger.name

        Returns: None

        """
        return rf.setLogger(self, get_new_logger=get_new_logger, or_list=or_list, and_list=and_list)

    def setLoggerField(self, field_name: str, new_field_value: object, or_list: List[str] = [], and_list: List[str] = []):
        """
        Sets all selected logger fields to new value.
        So for each logger_instance: logger_instance.field_name = new_field_value
        If the field is a tf.Variable the assign method is used instead.
        Args:
            field_name: The name of the field.
            new_field_value: The new value.
            or_list: at least one of the regex have to match the logger.name
            and_list: all of the regex have to match the logger.name

        Returns: None

        """
        return rf.setLoggerField(self, field_name=field_name, new_field_value=new_field_value, or_list=or_list, and_list=and_list)

    def getQuantizers(self):
        """
         gets all quantizers.

         Returns: List of quantizers

         """
        return rf.getQuantizers(self)

    def getAllOfType(self, selected_type:T) -> List[T]:
        """
         gets all instances of a certain type.

         Returns: List of all instances in the model

         """
        return rf.getAllOfType(self, selected_type)

    def setQuantizer(self, get_new_quantizer: Callable[[object], object], or_list: List[str] = [], and_list: List[str] = []):
        """
        Sets all selected quantizers to a new quantizer instance. The instance is received from the get_new_quantizer function.

        Args:
            get_new_quantizer: a function that accepts the old quantizer instance and retruns a new one.
            IMPORTANT the new instance.name has to match the old instance.name
            or_list: at least one of the regex have to match the quantizer.name
            and_list: all of the regex have to match the quantizer.name

        Returns: None

        """
        return rf.setQuantizer(self, get_new_quantizer=get_new_quantizer, or_list=or_list, and_list=and_list)

    def setQuantizerField(self, field_name: str, new_field_value: object, or_list: List[str] = [], and_list: List[str] = []):
        """
        Sets all selected quantizer fields to new value.
        So for each quantizer_instance: quantizer_instance.field_name = new_field_value
        If the field is a tf.Variable the assign method is used instead.
        Args:
            field_name: The name of the field.
            new_field_value: The new value.
            or_list: at least one of the regex have to match the quantizer.name
            and_list: all of the regex have to match the quantizer.name

        Returns: None

        """
        return rf.setQuantizerField(self,field_name=field_name, new_field_value=new_field_value, or_list=or_list, and_list=and_list)

    def saveAllLoggers(self, folderpath):
        """
        Save and reset all loggers. All results are saved in the given folderpath
        Args:
            folderpath: The path to the folder to save the logs in.

        Returns: None

        """
        return rf.saveLoggers(self, folderpath=folderpath)

    def printQuantizerSelection(self, or_list: List[str] = [], and_list: List[str] = [], print_additional_info=lambda obj:"", print_path_structure=False):
        """
        Pretty prints the given selection of quantizers. Also, with the print_additional_info method, additional information
        for each entry can be displayed. like the name with print_additional_info=lambda obj: obj.name
        Args:
            or_list: at least one of the regex have to match the quantizer.name
            and_list: all of the regex have to match the quantizer.name
            print_additional_info:

        Returns: None

        """
        return rf.printQuantizerSelection(self, or_list=or_list, and_list=and_list, print_additional_info=print_additional_info, print_path_structure=print_path_structure)

    def printLoggerSelection(self, or_list: List[str] = [], and_list: List[str] = [], print_additional_info=lambda obj:"", print_path_structure=False):
        """
        Pretty prints the given selection of loggers. Also, with the print_additional_info method, additional information
        for each entry can be displayed. like the name with print_additional_info=lambda obj: obj.name
        Args:
            or_list: at least one of the regex have to match the logger.name
            and_list: all of the regex have to match the logger.name
            print_additional_info:

        Returns: None

        """
        return rf.printLoggerSelection(self, or_list=or_list, and_list=and_list, print_additional_info=print_additional_info, print_path_structure=print_path_structure)