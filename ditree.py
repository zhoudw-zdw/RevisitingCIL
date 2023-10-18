import os

def generate_dirtree(path, indent=0):
    dirtree = ""
    for item in os.listdir(path):
        if 'configs' in path or 'data' in path or 'logs' in path or 'checkpoints' in path or '.git' in path or 'pycache' in path:
            continue
        item_path = os.path.join(path, item)
        if os.path.isdir(item_path):
            dirtree += "\n" + ".{} {}".format(indent+2, item) + "."
            dirtree += generate_dirtree(item_path, indent + 1)
        else:
            dirtree += "\n" + ".{} {}".format(indent+2, item) + "."
    return dirtree

dirtree = generate_dirtree('.')
latex_code = "\\dirtree{% " + dirtree + "\n}"

print(latex_code.replace('_', '\\_'))
