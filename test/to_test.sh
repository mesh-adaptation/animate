#!/usr/bin/bash

removestar -i $1
sed -i "s/^/    /" $1
sed -i "1idef test_demo():" $1
sed -i "1ios.environ['ANIMATE_REGRESSION_TEST'] = '1'" $1
sed -i "1iimport os" $1
