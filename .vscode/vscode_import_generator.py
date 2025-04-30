# bazel build .vscode:all && ./bazel-bin/.vscode/vscode_import_generator
# https://github.com/microsoft/pylance-release/discussions/2712
import json
import os

BAZEL_WORKSPACE = "template_py"
BAZEL_PATH = f"bazel-{BAZEL_WORKSPACE}/external"
RULES_PYTHON_PREFIXS = "rules_python++pip+pip_"
VSCODE_BAZEL_PATH = "${{workspaceFolder}}/{}".format(BAZEL_PATH)
VSCODE_SETTINGS_FILE = ".vscode/settings.json"


def main():
    def check_word_start_with_right_prefix(word):
        return word.startswith(RULES_PYTHON_PREFIXS)

    def add_prefix(word):
        return os.path.join(VSCODE_BAZEL_PATH, word)

    bazel_externals = os.listdir(BAZEL_PATH)
    filtered_list = [s for s in bazel_externals if check_word_start_with_right_prefix(s)]
    filtered_list = [add_prefix(s) for s in filtered_list]
    filtered_list.sort()

    with open(VSCODE_SETTINGS_FILE, "r") as f:
        vscode_settings = json.loads(f.read())

    vscode_settings["python.analysis.extraPaths"] = filtered_list
    vscode_settings["python.autoComplete.extraPaths"] = filtered_list

    with open(VSCODE_SETTINGS_FILE, "w") as f:
        f.write(json.dumps(vscode_settings, indent=4, sort_keys=True))


if __name__ == "__main__":
    main()
