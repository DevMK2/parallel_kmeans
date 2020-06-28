SHELL = /bin/bash

SUB_DIRS = mnist kmeans 

default_target: subdirs 

subdirs:
	@for dir in $(SUB_DIRS); do\
		$(MAKE) -C $$dir all;\
		if [ $$? != 0 ]; then exit 1; fi;\
	done

cleanall :
	@for dir in $(SUB_DIRS); do\
		$(MAKE) -C $$dir clean;\
		if [ $$? != 0 ]; then exit 1; fi;\
	done
