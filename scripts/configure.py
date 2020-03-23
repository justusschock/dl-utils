import os
import shutil


def get_root_path():
    return os.path.dirname(os.path.dirname(__file__))


def set_repo_name(name: str):
    path = get_root_path()
    new_path = os.path.join(os.path.dirname(path), name)
    shutil.move(path, new_path)

    return new_path


def read_file(file: str):
    with open(file) as f:
        content = f.readlines()
    return content


def remove_script(new_path: str):
    os.remove(os.path.join(new_path,
                           os.path.basename(os.path.dirname(__file__)),
                           os.path.basename(__file__)))


def replace_file_content(file: str, original_content: str, new_content: str):
    curr_content = read_file(file)

    for idx, line in enumerate(curr_content):
        curr_content[idx] = line.replace(original_content, new_content)

    with open(file, 'w') as f:
        f.writelines(curr_content)


def remove_starting_string(file: str, start_str: str):
    curr_content = read_file(file)

    for idx, line in enumerate(curr_content):
        if line.startswith(start_str):
            curr_content[idx] = line[len(start_str):]

    with open(file, 'w') as f:
        f.writelines(curr_content)


def remove_lines_with_starting_string(file: str, start_str: str):
    curr_content = read_file(file)

    new_content = []
    for line in curr_content:
        if line.startswith(start_str):
            continue

        new_content.append(line)

    with open(file, 'w') as f:
        f.writelines(new_content)


def set_package_name(name: str):
    files = ['setup.cfg', 'setup.py',
             os.path.join('.github', 'workflows', 'unittests.yml')]

    base_path = get_root_path()

    for file in files:
        replace_file_content(os.path.join(base_path, file), 'REPONAME', name)

    shutil.move(os.path.join(base_path, 'REPONAME'),
                os.path.join(base_path, name))


def configure_coverage_upload(enable: bool):
    file = os.path.join(get_root_path(),
                        '.github', 'workflows', 'unittests.yml')
    start_str = '#COVERAGE_UPLOAD_CONFIG'

    if enable:
        remove_starting_string(file, start_str=start_str)
    else:
        remove_lines_with_starting_string(file, start_str=start_str)


def exec_fn_with_exception_guard(fn, *args, **kwargs):
    ret_val = None
    try:
        ret_val = fn(*args, **kwargs)
        return True, ret_val
    except Exception as e:
        print(e)
        return False, ret_val


def successfull(*args, **kwargs):
    return True


if __name__ == '__main__':
    from functools import partial
    import argparse
    from collections import OrderedDict

    parser = argparse.ArgumentParser()
    parser.add_argument('--repo_name', type=str,
                        help='The new folder name for the repository',
                        default=None)
    parser.add_argument(
        '--package_name', type=str,
        help='The new name of the python package included in this directory',
        default=None)

    parser.add_argument('--enable_coverage_upload', action='store_true',
                        help='Whether to enable or disable the coverage '
                             'reports for this repository')

    parser.add_argument(
        '--remove_script', action='store_true',
        help='Whether or not to remove this script after success')

    parser_args = parser.parse_args()

    returns, successes = {}, {}
    functions = OrderedDict()

    if parser_args.repo_name is None:
        repo_name_fn = successfull
    else:
        repo_name_fn = partial(set_repo_name, name=parser_args.repo_name)

    if parser_args.package_name is None:
        package_name_fn = successfull
    else:
        package_name_fn = partial(set_package_name,
                                  name=parser_args.package_name)

    functions['coverage_upload'] = partial(
        configure_coverage_upload,
        enable=parser_args.enable_coverage_upload)

    functions['package_name'] = package_name_fn
    functions['repo_name'] = repo_name_fn

    for name, func in functions.items():
        success, ret_val = exec_fn_with_exception_guard(func)

        returns[name] = ret_val
        successes[name] = success

    if parser_args.remove_script:
        remove_script(returns['repo_name'])
