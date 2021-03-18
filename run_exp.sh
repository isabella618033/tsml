#!/bin/bash                                                                                                                                                                                                       
date                                                                                                                                                                                                              
java -Xms128m -Xmx8g -Dfile.encoding=UTF-8 -classpath build/libs/tsml-all-0.1.0.jar runExperiment.runExperiment
