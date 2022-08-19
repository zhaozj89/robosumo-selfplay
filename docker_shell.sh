#!/usr/bin/env bash

if hash nvidia-docker 2>/dev/null; then
  cmd=nvidia-docker
else
  cmd=docker
fi

${cmd} run --rm --user $(id -u) -v `pwd`:/home/$USER/selfplay -it selfplay:1.0