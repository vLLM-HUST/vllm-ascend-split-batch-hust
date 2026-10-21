#!/usr/bin/env bash
# 幂等修复 login 环境的 LD_LIBRARY_PATH：
#   /etc/profile 三个 set_env 拼好的路径会被 workspace hook
#   （~/.profile -> /etc/profile.d/02-workspace-user-env.sh）按
#   /root/.ssh/environment 里的旧值覆盖（诊断: records/probes/20261021T024500Z）。
#   本脚本把三个 set_env 的合成值写回该文件（hook 的输入, 「用户配置优先」），
#   只改 LD_LIBRARY_PATH 一行, 其余行逐字节保留; 旧值中多出的条目按原序追加。
#   回滚: 用脚本打印的 .bak-<ts> 覆盖回去即可。
set -uo pipefail

usage() {
  echo "usage: $0 [--env-file <path>] [--value <ld_library_path>] [--dry-run]" >&2
  exit 2
}

env_file=/root/.ssh/environment
value="" dry=0
while [ $# -gt 0 ]; do
  case "$1" in
    --env-file) env_file=${2:?--env-file 需路径}; shift 2 ;;
    --value) value=${2:?--value 需 LD_LIBRARY_PATH 值}; shift 2 ;;
    --dry-run) dry=1; shift ;;
    *) usage ;;
  esac
done

derive() {
  local f
  for f in /usr/local/Ascend/ascend-toolkit/set_env.sh \
    /usr/local/Ascend/cann-9.1.0/share/info/ascendnpu-ir/bin/set_env.sh \
    /usr/local/Ascend/nnal/atb/set_env.sh; do
    [ -r "$f" ] || {
      echo "repair-ascend-login-env: 缺 $f, 请用 --value 显式给值" >&2
      return 1
    }
  done
  bash -c '
    source /usr/local/Ascend/ascend-toolkit/set_env.sh >/dev/null 2>&1
    source /usr/local/Ascend/cann-9.1.0/share/info/ascendnpu-ir/bin/set_env.sh >/dev/null 2>&1
    source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=1 >/dev/null 2>&1
    printf "%s" "$LD_LIBRARY_PATH"'
}

merge_extras() { # $1=base(权威合成值) $2=extras(旧值) -> 两段按序拼接、同值去重保首现、丢空段
  local merged="" entry rest
  for rest in "$1" "$2"; do
    while [ -n "$rest" ]; do
      entry=${rest%%:*}
      rest=${rest#*:}
      [ "$rest" = "$entry" ] && rest=""
      if [ -n "$entry" ]; then
        case ":${merged}:" in
          *":${entry}:"*) ;;
          *) merged="${merged:+$merged:}$entry" ;;
        esac
      fi
    done
  done
  printf '%s' "$merged"
}

[ -r "$env_file" ] || { echo "repair-ascend-login-env: $env_file 不存在" >&2; exit 1; }
old_value=$(grep '^LD_LIBRARY_PATH=' "$env_file" | tail -1 | cut -d= -f2-)
[ -n "$value" ] || value=$(derive) || exit 1
merged=$(merge_extras "$value" "$old_value")
if [ "$merged" = "$old_value" ]; then
  echo "repair-ascend-login-env: 已一致, 不动 $env_file"
  exit 0
fi

tmp=$(mktemp)
awk -v line="LD_LIBRARY_PATH=$merged" '
  /^LD_LIBRARY_PATH=/ && !done { print line; done = 1; next }
  { print }
  END { if (!done) print line }' "$env_file" >"$tmp"

echo "repair-ascend-login-env: 旧(${#old_value}B) -> 新(${#merged}B)"
if [ "$dry" = 1 ]; then
  echo "LD_LIBRARY_PATH=$merged"
  rm -f "$tmp"
  exit 0
fi

bak="${env_file}.bak-$(date +%Y%m%dT%H%M%SZ)"
cp -p "$env_file" "$bak"
cat "$tmp" >"$env_file"
rm -f "$tmp"
echo "repair-ascend-login-env: 已写入 $env_file (备份: $bak)"
echo "  回滚: cp -p $bak $env_file"
echo "  before sha256: $(sha256sum "$bak" | cut -d' ' -f1)"
echo "  after  sha256: $(sha256sum "$env_file" | cut -d' ' -f1)"
