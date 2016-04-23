# Set the task name
TASK = msf_trending

# Uncomment the correct choice indicating either SKA or TST flight environment
# FLIGHT_ENV = SKA
# FLIGHT_ENV = TST

# Set the names of all files that get installed
#  Examples for celmon
#  TASK = celmon
#  BIN = celmon.pl 
#  SHARE = calc_offset.pl
#  DATA = CELMON_table.rdb ICRS_tables
WEB = index.html
SHARE = msf_trending.py
DATA = task_schedule.cfg msd_data.dat ref_obs_data.dat
DOC = 

include /proj/sot/ska/include/Makefile.FLIGHT

# Define outside data and bin dependencies required for testing,
# i.e. all tools and data required by the task which are NOT 
# created by or internal to the task itself.  These will be copied
# from the ROOT_FLIGHT area.
#
# TEST_DEP = bin/skycoor data/EPHEM/gephem.dat
TEST_DEP = 

# To 'test', first check that the INSTALL root is not the same as the FLIGHT root
# with 'check_install' (defined in Makefile.FLIGHT).  Typically this means doing
#  setenv TST $PWD
# Then copy any outside data or bin dependencies into local directory via
# dependency rules defined in Makefile.FLIGHT

# Testing no long creates a lib/perl link, since Perl should find the library
# because perlska puts /proj/sot/ska/lib/perl (hardwired) into PERL5LIB.

#test: check_install $(BIN) $(TEST_DEP) install
#	$(INSTALL_BIN)/task.pl

# Make a versioned distribution.  Could also use an EXCLUDE_MANIFEST
#dist:
#	mkdir $(NAME)-$(VERSION)
#	tar --exclude CVS --exclude "*~" --create --files-from=MANIFEST --file - \
#	 | (tar --extract --directory $(NAME)-$(VERSION) --file - )
#	tar --create --verbose --gzip --file $(NAME)-$(VERSION).tar.gz $(NAME)-$(VERSION)
#	rm -rf $(NAME)-$(VERSION)



install:
#  Uncomment the lines which apply for this task
#	mkdir -p $(INSTALL_BIN)
	mkdir -p $(INSTALL)/www/ASPECT/$(TASK)
	mkdir -p $(INSTALL_DATA)
	mkdir -p $(INSTALL_SHARE)
#	mkdir -p $(INSTALL_DOC)
#	mkdir -p $(INSTALL_LIB)
#	rsync --times --cvs-exclude $(BIN) $(INSTALL_BIN)/
	rsync --times --cvs-exclude $(DATA) $(INSTALL_DATA)/
	rsync --times --cvs-exclude $(SHARE) $(INSTALL_SHARE)/
	rsync --times --cvs-exclude $(WEB) $(INSTALL)/www/ASPECT/$(TASK)
#	rsync --times --cvs-exclude $(DOC) $(INSTALL_DOC)/
#	rsync --times --cvs-exclude $(LIB) $(INSTALL_LIB)/
#	pod2html task.pl > $(INSTALL_DOC)/task.html
