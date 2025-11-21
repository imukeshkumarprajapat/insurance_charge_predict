import sys   #error ko track karne me mdd karta h ye 
from src.insurance_charge_predict.logger import logging



import sys
from src.insurance_charge_predict.logger import logging

def error_message_detail(error,error_detail:sys):
    #_,_,exc_tb=error_detail.exc_info()
    exc_type, exc_value, exc_tb = sys.exc_info() 
    file_name=exc_tb.tb_frame.f_code.co_filename
    error_message="Error occured in python script name [{0}] line number [{1}] error message[{2}]".format(
     file_name,exc_tb.tb_lineno,str(error))

    return error_message


# class CustomeException(Exception):
#     def __init__(self,error_message,error_details:sys):
#         super().__init__(error_message)
#         self.error_message=error_message_detail(error_message,error_details)

#     def __str__(self):
#         return self.error_message
import sys

import sys

class CustomeException(Exception):
    def __init__(self, error, error_detail: sys):
        super().__init__(str(error))
        self.error_message = CustomeException.error_message_detail(error)

    @staticmethod
    def error_message_detail(error):
        exc_type, exc_value, exc_tb = sys.exc_info()   # ✅ traceback details
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
        return f"Error occurred in python script name [{file_name}] line number [{line_number}] error message [{str(error)}]"