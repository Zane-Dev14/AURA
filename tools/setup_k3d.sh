#!/bin/bash
set -e
k3d cluster create aura --agents 2 --servers 1 -p "8080:80@loadbalancer"
