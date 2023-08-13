#!/bin/bash

zeroStage=3

if [ "$zeroStage" -eq 3 ];then   # true == zero3, false == zero1
    echo "3"

else
  if [ "$zeroStage" -eq 0 ];then
    echo "0"
  elif [ "$zeroStage" -eq 1 ];then
    echo "1"
  fi
fi