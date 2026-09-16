#!/usr/bin/env bash
set -e

PY_VERSION=$1

apt update -y

apt install -y \
  build-essential libssl-dev zlib1g-dev libbz2-dev \
  libreadline-dev libsqlite3-dev wget curl llvm \
  libncurses5-dev libncursesw5-dev xz-utils tk-dev \
  libffi-dev liblzma-dev git make

echo "=== Installing pyenv ==="
if [ ! -d "$HOME/.pyenv" ]; then
  curl https://pyenv.run | bash
else
  echo "pyenv is already installed, skipping installation."
fi

if ! grep -q 'pyenv init' ~/.bashrc; then
  echo 'export PATH="$HOME/.pyenv/bin:$PATH"' >> ~/.bashrc
  echo 'eval "$(pyenv init --path)"' >> ~/.bashrc
  echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc
fi

export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv virtualenv-init -)"

# Install the oldest PY version for the given major.minor release
ACTUAL_PY_VERSION=$(pyenv install --list | grep -E "^\s*$PY_VERSION\.[0-9]+$" | head -1 | tr -d ' ')

echo "=== Installing Python $ACTUAL_PY_VERSION (it may take a while...) ==="
pyenv install -s $ACTUAL_PY_VERSION

echo "=== Setting Python $ACTUAL_PY_VERSION as default ==="
pyenv global $ACTUAL_PY_VERSION

echo "=== Done! ==="