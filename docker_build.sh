#!/usr/bin/env bash
docker build --build-arg UID=$UID --build-arg USER=$USER -t selfplay:1.0 .