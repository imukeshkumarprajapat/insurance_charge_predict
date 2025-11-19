from setuptools import find_packages,setup
from typing import List  #retrun list 

def get_requirements(file_path:str)->List[str]:
    """
    this function will retrun the list of requirements"""

    requirements=[]
    
    with open(file_path) as file_obj: #file ko open karna
        requirements=file_obj.readlines() #line by line file read karna
        requirements= [req.replace('\n',"") for req in requirements] #\n new line remove karna

    return requirements








setup(
    name="insurance_price_pridict",
    version='0.0.1',
    auther="mukesh",
    author_email="www.worldwide.com",
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')#["pandas","numpy"]
)