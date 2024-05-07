#!/bin/bash

for d in $(find ~+ -name "modfiles" -type d); do
    cd ${d}/..
    pwd
    nrnivmodl modfiles
done