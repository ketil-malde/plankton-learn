# Change the configuration here.
# Include your useid/name as part of IMAGENAME to avoid conflicts
IMAGENAME = plankton-learn
COMMAND   = bash
DISKS     = -v $(PWD)/../plankton-data:/data:ro -v $(PWD):/project
PORT      =
# For jupyter:
# PORT    = -p 8888:8888
NETWORK =
# Sometimes necessary for networking to work:
# NETWORK   = --network host
GPU       = 0
RUNTIME   = --gpus device=$(GPU)
#RUNTIME   =

# No need to change anything below this line
USERID    = $(shell id -u)
GROUPID   = $(shell id -g)
USERNAME  = $(shell whoami)

# Allows you to use sshfs to mount disks
SSHFSOPTIONS = --cap-add SYS_ADMIN --device /dev/fuse

USERCONFIG   = --build-arg user=$(USERNAME) --build-arg uid=$(USERID) --build-arg gid=$(GROUPID)

.PHONY: .docker test train

.docker: Dockerfile
	docker build $(USERCONFIG) $(NETWORK) -t $(USERNAME)-$(IMAGENAME) -f Dockerfile .

# WEIGHTS = mask_rcnn_coco.h5

# Using -it for interactive use
RUNCMD=docker run $(RUNTIME) $(NETWORK) --rm --user $(USERID):$(GROUPID) $(PORT) $(SSHFSOPTIONS) $(DISKS) -it $(USERNAME)-$(IMAGENAME)

# Starts and interactive shell by default (COMMAND = bash)
default: .docker
	$(RUNCMD) $(COMMAND)

# $(WEIGHTS): src/download_weights.py
# 	$(RUNCMD) python3 src/download_weights.py

train: .docker
	$(RUNCMD) python3 /src/train.py

test: .docker
	$(RUNCMD) python3 /src/test.py
