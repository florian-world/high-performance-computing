#!/usr/bin/zsh

ncomp=('2' '3' '4' '5')
nbatch=('10' '16' '20' '32' '40' '64')


for j in $nbatch; do
  for i in $ncomp; do
      echo "oja/build/Desktop-Release/oja $i $j"
      oja/build/Desktop-Release/oja $i $j
  done
done