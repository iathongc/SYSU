#!/bin/bash

# 编译
mpicc -O3 -march=native -o heated_plate_mpi heated_plate_mpi.c

if [ $? -ne 0 ]; then
  echo "Compilation failed."
  exit 1
fi

echo "Compilation successful."

# 运行
mpirun -np 4 ./heated_plate_mpi
if [ $? -ne 0 ]; then
  echo "Execution failed."
  exit 1
fi

echo "Execution completed successfully."

