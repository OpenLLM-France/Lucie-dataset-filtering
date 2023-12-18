#!/usr/bin/python
# -*- coding: utf-8 -*-
""" 

@Author: Evan Dufraisse
@Date: Tue May 30 2023
@Contact: e[dot]dufraisse[at]gmail[dot]com
@License: MIT License
"""
import os


if __name__ == "__main__":
    root_folder_of_project = os.path.abspath(os.path.dirname(__file__))
    print("Root folder of project: {}".format(root_folder_of_project))

    # Ask for name of the project
    project_name = input("Name of the project: ")
    print("Project name: {}".format(project_name))

    # Ask for cli binding
    cli_binding_default = project_name.lower()
    cli_binding = input("CLI binding (default: {}): ".format(cli_binding_default))
    if cli_binding == "":
        cli_binding = cli_binding_default
    print("CLI binding: {}".format(cli_binding))

    # Ask for description
    description = input("Long Description: ")
    print("Description: {}".format(description))

    # Ask for short description
    short_description = input("Short Description: ")
    print("Short Description: {}".format(short_description))

    # Ask for python version
    python_version = input("Min python version (default: 3.8): ")
    if python_version == "":
        python_version = "3.8"

    # Ask for packages:
    packages = input("Packages (separated by ','): ")
    packages = packages.split(",")
    packages = [package.strip() for package in packages]

    with open(os.path.join(root_folder_of_project, "setup.py"), "r") as f:
        setup_file = f.read()

    setup_file = setup_file.replace("PROJECT_NAME", project_name)
    setup_file = setup_file.replace("CLI_BINDING", cli_binding)
    setup_file = setup_file.replace("DESCRIPTION", description)
    setup_file = setup_file.replace("PYTHON_VERSION", python_version)
    default_packages = ['"tqdm"', '"jsonlines"', '"loguru"', '"click"']
    set_already_added_packages = set([elem.strip('"') for elem in default_packages])
    packages_str = ",\n    ".join(['"tqdm"', '"jsonlines"', '"loguru"', '"click"'])
    packages_str = "\n    " + packages_str

    for pkg in packages:
        if pkg not in set_already_added_packages:
            packages_str += ',\n    "{}"'.format(pkg)
            set_already_added_packages.add(pkg)
    packages_str += "\n"

    setup_file = setup_file.replace("PACKAGES", packages_str)

    with open(os.path.join(root_folder_of_project, "setup.py"), "w") as f:
        f.write(setup_file)

    with open(os.path.join(root_folder_of_project, "README.md"), "r") as f:
        readme_file = f.read()

    readme_file = readme_file.replace("PROJECT_NAME", project_name)
    readme_file = readme_file.replace("SHORT_DESCRIPTION", short_description)
    readme_file = readme_file.replace("DESCRIPTION", description)
    readme_file = readme_file.replace("CLI_BINDING", cli_binding)

    with open(os.path.join(root_folder_of_project, "README.md"), "w") as f:
        f.write(readme_file)

    # Ask for default conda env
    conda_env = input("Default conda env (default: {}): ".format(cli_binding))
    with open(os.path.join(root_folder_of_project, "convert_jupyter.sh"), "r") as f:
        env_file = f.read()

    env_file = env_file.replace("CONDA_ENV", conda_env)
    env_file = env_file.replace("ABS_PATH", root_folder_of_project.rstrip("/"))

    with open(os.path.join(root_folder_of_project, "convert_jupyter.sh"), "w") as f:
        f.write(env_file)

    # Ask for creating data folder
    create_data_folder = input("Create data folder (default: y): ")
    if create_data_folder == "":
        create_data_folder = "y"
    if create_data_folder.lower() == "y":
        os.mkdir(os.path.join(os.environ["DATA_DIR"], project_name))
        print("Data folder created")
        os.system(
            "ln -s {} {}".format(
                os.path.join(os.environ["DATA_DIR"], project_name),
                os.path.join(root_folder_of_project, "data"),
            )
        )

    # Chang name folder

    os.rename(
        os.path.join(root_folder_of_project, "src", "CLI_BINDING"),
        os.path.join(root_folder_of_project, "src", cli_binding),
    )

    # Change cli.py

    with open(
        os.path.join(root_folder_of_project, "src", cli_binding, "cli.py"), "r"
    ) as f:
        cli_file = f.read()

    cli_file = cli_file.replace("CLI_BINDING", cli_binding)
    with open(
        os.path.join(root_folder_of_project, "src", cli_binding, "cli.py"), "w"
    ) as f:
        f.write(cli_file)

    # Init git

    print("Initializing git...")
    os.system("git init")
    os.system("git add *")
    os.system('git commit -m "Initial commit"')
