import os

files = [
    r"c:\meta\inference.py",
    r"c:\meta\customer_support_env\graders\medium_grader.py",
    r"c:\meta\customer_support_env\graders\hard_grader.py",
    r"c:\meta\customer_support_env\environment.py",
    r"c:\meta\customer_support_env\graders\easy_grader.py"
]

for f in files:
    with open(f, "r", encoding="utf-8") as file:
        content = file.read()
        
    content = content.replace("1 - 0.001", "0.9999")
    content = content.replace("0.001", "0.0001")
    
    with open(f, "w", encoding="utf-8") as file:
        file.write(content)
        
print("Replacement successful.")
