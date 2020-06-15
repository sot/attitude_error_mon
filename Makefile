# Set the task name
TASK = attitude_error_mon

SHARE = att_err_mon.py index_template.html
DATA = task_schedule.cfg 

SKA3 = $(SKA)
INSTALL_DATA = $(SKA3)/data/$(TASK)
INSTALL_SHARE = $(SKA3)/share/$(TASK)


install:
#  Uncomment the lines which apply for this task
	mkdir -p $(SKA3)/www/ASPECT/$(TASK)
	mkdir -p $(INSTALL_DATA)
	mkdir -p $(INSTALL_SHARE)
	rsync --times --cvs-exclude $(DATA) $(INSTALL_DATA)/
	rsync --times --cvs-exclude $(SHARE) $(INSTALL_SHARE)/


