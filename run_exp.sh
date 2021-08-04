#!/bin/bash
date
# java -Xms128m -Xmx8g -Dfile.encoding=UTF-8 -classpath build/libs/tsml-exp.jar runExperiment.runExperiment run_exp.config
java -Xms128m -Xmx10g -Dfile.encoding=UTF-8 -classpath "./lib/tsml-exp.jar:./lib/*" runExperiment.runExperiment run_exp.config
