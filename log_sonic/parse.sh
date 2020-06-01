#!/bin/bash

grep 'Client time' $1 | grep -o '[0-9]*' | cat >> parsed_data/$1
