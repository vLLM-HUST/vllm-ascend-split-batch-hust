#!/bin/bash
# Source this before serving: `. tools/serve-env.sh`
#
# Unlike build-env.sh this does NOT prepend the build venv: serving runs vllm
# from the interpreter pinned as locations.serving_python, and mixing that
# venv's torch with the serving interpreter's vllm is exactly the mismatch the
# manifest exists to prevent.
#
# STACK_MANIFEST selects a deployment snapshot manifest; unset keeps the repo
# manifest.  Resolution failure aborts the source: continuing would apply an
# empty export set and then source vendor scripts from unset paths.
#
# Neither vendor set_env.sh is nounset-clean: ATB reads $ZSH_VERSION and
# /etc/bash.bashrc reads unguarded $PS1. Keep -u off across both sources.

_stack_repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
_stack_python="${STACK_PYTHON:-python3}"
_stack_manifest="${STACK_MANIFEST:-}"

_saved_opts=$(set +o)
set +u
set +e
if [ -n "${_stack_manifest}" ]; then
    _stack_exports=$("${_stack_python}" "${_stack_repo}/tools/stack_versions.py" --manifest "${_stack_manifest}" --serve-env)
else
    _stack_exports=$("${_stack_python}" "${_stack_repo}/tools/stack_versions.py" --serve-env)
fi
_stack_rc=$?
if [ "${_stack_rc}" -ne 0 ] || [ -z "${_stack_exports}" ]; then
    echo "serve-env: manifest resolution failed (rc=${_stack_rc}, manifest=${_stack_manifest:-repo default})" >&2
    eval "${_saved_opts}"
    unset _saved_opts _stack_python _stack_manifest _stack_exports _stack_rc _stack_repo
    return 1
fi
eval "${_stack_exports}"
if [ -z "${ASCEND_HOME_PATH}" ] || [ ! -f "${ASCEND_HOME_PATH}/set_env.sh" ]; then
    echo "serve-env: no CANN set_env.sh under ASCEND_HOME_PATH (${ASCEND_HOME_PATH:-unset})" >&2
    eval "${_saved_opts}"
    unset _saved_opts _stack_python _stack_manifest _stack_exports _stack_rc _stack_repo
    return 1
fi
. "${ASCEND_HOME_PATH}/set_env.sh"
if [ -f "${ATB_ENV}" ]; then
    . "${ATB_ENV}" --cxx_abi="${ATB_CXX_ABI}"
fi
eval "${_saved_opts}"
unset _saved_opts _stack_python _stack_manifest _stack_exports _stack_rc _stack_repo
