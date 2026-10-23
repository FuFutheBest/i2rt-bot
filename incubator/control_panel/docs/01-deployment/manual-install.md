# Manual installation / 传统手动部署

本流程与 Codex Prompt 使用相同的校验门槛。请在目标 Linux 主机的新 Bash shell
中执行。

本流程同样是 **release-only**：只能下载并启动指定 GitHub Release 的归档。禁止用
源码 checkout、Windows/WSL 工作区、旧 conda/venv、系统 Python 或在线 `pip install`
补齐或替代归档内容。Release 附件缺失或任一校验失败时必须停止，不得退回源码部署。

## 1. Set release and deployment values / 设置版本与路径

从 GitHub release URL 读取仓库和 tag，并替换下列占位符：

```bash
set -euo pipefail

export RELEASE_URL='<RELEASE_URL>'
export RELEASE_REPO='<OWNER/REPOSITORY>'
export RELEASE_TAG='<RELEASE_TAG>'
export DEPLOY_ROOT='<DEPLOY_ROOT>'
export PORT='<PORT>'
export SUPERINFERENCE_DATA_ROOT='<DATA_ROOT>'
```

仓库示例为 `NoematrixTeam/SuperInference`。不要从其他 release 复制 tag、
commit、归档名或 checksum。

## 2. Check the host and GitHub access / 检查目标机与 GitHub 权限

```bash
uname -m
lsb_release -ds
ldd --version 2>&1 | sed -n '1p'
df -h "$(dirname "$DEPLOY_ROOT")"
# `gh auth status` reports stale inactive accounts as a non-zero exit even when
# the active account is valid. Display the diagnostics, then verify the active
# account with an authenticated API call.
gh auth status --hostname github.com || true
gh api user --jq .login
gh release view "$RELEASE_TAG" --repo "$RELEASE_REPO" \
  --json tagName,targetCommitish,url,assets
```

目标平台必须是 Ubuntu 22.04+ x86_64、glibc 2.35+。如果 `gh` 尚未获得私有仓库
权限，使用官方 device flow：

```bash
gh auth login --hostname github.com --git-protocol ssh --web
```

Git SSH 与 release 下载是两套鉴权：即使源码使用 SSH key，
`gh release download` 仍通过已登录的 GitHub HTTPS API。

## 3. Download the assets / 下载附件

必须使用当前 tag 专属的空目录，避免 glob 误选其他版本文件：

```bash
mkdir -p "$DEPLOY_ROOT/downloads/$RELEASE_TAG" \
  "$DEPLOY_ROOT/releases" "$SUPERINFERENCE_DATA_ROOT" \
  "$DEPLOY_ROOT/logs" "$DEPLOY_ROOT/run"

gh release download "$RELEASE_TAG" --repo "$RELEASE_REPO" \
  --dir "$DEPLOY_ROOT/downloads/$RELEASE_TAG"

cd "$DEPLOY_ROOT/downloads/$RELEASE_TAG"
```

检查附件列表。常规分片 release 包含一个分片 checksum、一个整包 checksum，
以及连续编号的分片：

```bash
find . -maxdepth 1 -type f -printf '%f\n' | sort
```

只允许选中一套“分片”或“单文件”归档，然后校验。下面的检查会在候选不唯一时
主动失败：

```bash
shopt -s nullglob
part_checksums=(./parts-*.sha256)
archive_checksums=(./superinference-openpi-*.tar.zst.sha256)
part_assets=(./*.tar.zst.part-*)
direct_archives=(./superinference-openpi-*.tar.zst)

test "${#archive_checksums[@]}" -eq 1
archive_checksum="${archive_checksums[0]}"
archive="${archive_checksum%.sha256}"

if test "${#part_assets[@]}" -gt 0; then
  test "${#part_checksums[@]}" -eq 1
  test "${#direct_archives[@]}" -eq 0
  sha256sum -c "${part_checksums[0]}"
  mapfile -t ordered_parts < <(printf '%s\n' "${part_assets[@]}" | sort)
  for index in "${!ordered_parts[@]}"; do
    printf -v expected_suffix '.part-%02d' "$index"
    test "${ordered_parts[$index]}" = "$archive$expected_suffix"
  done
  cat "${ordered_parts[@]}" > "$archive"
else
  test "${#part_checksums[@]}" -eq 0
  test "${#direct_archives[@]}" -eq 1
  test "${direct_archives[0]}" = "$archive"
fi

sha256sum -c "$archive_checksum"
zstd -t "$archive"
```

如果 glob 没有匹配、出现多个 checksum、分片不是从 `00` 连续编号，或任一校验
失败，应立即停止。

## 4. Extract and validate / 解压与校验

```bash
archive_listing="$DEPLOY_ROOT/downloads/$RELEASE_TAG/archive-files.txt"
tar --zstd -tf "$archive" > "$archive_listing"
archive_root="$(awk -F/ 'NF { print $1; exit }' "$archive_listing")"

test -n "$archive_root"
case "$archive_root" in
  superinference-openpi-*-linux-x86_64) ;;
  *) echo "Unexpected archive root: $archive_root" >&2; exit 1 ;;
esac
release_dir="$DEPLOY_ROOT/releases/$archive_root"
test ! -e "$release_dir"

tar --zstd -xf "$archive" -C "$DEPLOY_ROOT/releases"
cd "$release_dir"
sha256sum -c manifest/SHA256SUMS
./bin/superinference-doctor
```

将 release tag 解析为完整 commit，并与 manifest 比较。不要事后解析可能已经前移
的 target branch：

```bash
target_commitish="$(gh release view "$RELEASE_TAG" --repo "$RELEASE_REPO" \
  --json targetCommitish --jq .targetCommitish)"
tag_sha="$(gh api "repos/$RELEASE_REPO/commits/$RELEASE_TAG" --jq .sha)"

jq -e --arg sha "$tag_sha" \
  '.git_commit == $sha and (.git_status | length) == 0
    and .target == "linux-x86_64"' \
  manifest/release.json

if [[ "$target_commitish" =~ ^[0-9a-fA-F]{40}$ ]]; then
  test "${target_commitish,,}" = "${tag_sha,,}"
fi
```

归档 checksum、内部 checksum、manifest 比对或 doctor 任一失败，都不能启动。

## 5. Activate and start / 激活并启动

只有全部验证通过后才更新 `current`：

```bash
if ss -H -ltn "sport = :$PORT" | grep -q .; then
  echo "Port $PORT is already in use; inspect its owner before continuing." >&2
  ss -H -ltnp "sport = :$PORT" >&2 || true
  exit 1
fi

next_link="$DEPLOY_ROOT/.current.$$.new"
ln -s "$release_dir" "$next_link"
mv -Tf "$next_link" "$DEPLOY_ROOT/current"

log="$DEPLOY_ROOT/logs/control-panel-${RELEASE_TAG}.log"
pid_file="$DEPLOY_ROOT/run/control-panel-${RELEASE_TAG}.pid"

nohup env \
  SUPERINFERENCE_DATA_ROOT="$SUPERINFERENCE_DATA_ROOT" \
  SUPERINFERENCE_CONTROL_PANEL_HOST=0.0.0.0 \
  SUPERINFERENCE_CONTROL_PANEL_PORT="$PORT" \
  "$DEPLOY_ROOT/current/start_server.sh" \
  >"$log" 2>&1 </dev/null &

echo "$!" > "$pid_file"
```

如果端口已占用，不要 kill 未知进程。必须先从 PID、命令行和工作目录共同确认它
属于该部署根目录。

## 6. Read-only smoke check / 只读冒烟检查

```bash
curl --fail --show-error "http://127.0.0.1:$PORT/" >/dev/null
curl --fail --show-error "http://127.0.0.1:$PORT/recordings" >/dev/null
curl --fail --show-error "http://127.0.0.1:$PORT/api/state" | jq .
```

这些检查不得启动硬件。全部通过后再打开 `http://<TARGET_HOST>:$PORT/`。
