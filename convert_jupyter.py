import os


def convert_jupyter(path):
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".ipynb") and not file.endswith(".nbconvert.ipynb"):
                os.system(
                    "jupyter nbconvert --ClearOutputPreprocessor.enabled=True --to notebook "
                    + os.path.join(root, file)
                )


if __name__ == "__main__":
    convert_jupyter(".")