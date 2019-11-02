#!/usr/bin/env bash
sbatch --nodes=1 --time=5:00:00 --mem=32G --cpus=4 --gres=gpu:1 CDsVinyl-s-e-cde-vae.sh


