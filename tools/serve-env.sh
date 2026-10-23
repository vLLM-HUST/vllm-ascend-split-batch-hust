#!/bin/bash
# Source this before serving: `. tools/serve-env.sh`
#
# Unlike build-env.sh this does NOT prepend the build venv: serving runs vllm
# from the interpreter pinned as locations.serving_python, and mixing that
# venv's torch with the serving interpreter's vllm is exactly the mismatch the
# manifest exists to prevent.
#
# Neither vendor set_env.sh is nounset-clean: ATB reads $ZSH_VERSION and
# /etc/bash.bashrc reads unguarded $PS1. Keep -u off across both sources.

_stack_repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
_stack_python="${STACK_PYTHON:-python3}"

_saved_opts=$(set +o)
set +u
set +e
eval "$("${_stack_python}" "${_stack_repo}/tools/stack_versions.py" --serve-env)"
. "${ASCEND_HOME_PATH}/set_env.sh"
if [ -f "${ATB_ENV}" ]; then
    . "${ATB_ENV}" --cxx_abi="${ATB_CXX_ABI}"
fi
eval "${_saved_opts}"
unset _saved_opts _stack_python
